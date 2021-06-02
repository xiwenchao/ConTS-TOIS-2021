# BB-8 and R2-D2 are best friends.

import sys
import time
from collections import defaultdict
import random

random.seed(0)
sys.path.insert(0, '../FM')
sys.path.insert(0, '../yelp')

import json
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_value_, clip_grad_norm_
from torch.distributions import Categorical

from message import message
from config_3 import global_config as cfg
from utils_entropy import cal_ent
from heapq import nlargest, nsmallest
from utils_fea_sim_3 import feature_similarity
from utils_fea_sim_3 import feature_similarity_micro
from utils_sense_3 import try_feature_cause_change, rank_items
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
import math

r_list = [-1, -0.3, -1.5, 1.5, 1.2]# quit by user ,fail to recommend, ask attribute not accepted, ask attribute accepted, accept recommendation
d = 64
v = 0.01


def cuda_(var):
    return var.cuda() if torch.cuda.is_available() else var


class agent():
    def __init__(self, FM_model, user_id, busi_id, do_random, write_fp, strategy, TopKTaxo,
                 PN_model, log_prob_list, action_tracker, candidate_length_tracker, mini,
                 optimizer1_fm, optimizer2_fm, alwaysupdate, epsilon, sample_dict, choose_pool):
        # _______ input parameters_______
        self.user_id = user_id
        self.busi_id = busi_id
        self.FM_model = FM_model
        self.bias = list()

        self.turn_count = 0
        self.F_dict = defaultdict(lambda: defaultdict())
        self.recent_candidate_list = [int(k) for k, v in cfg.item_dict.items()]
        self.recent_candidate_list_ranked = self.recent_candidate_list  # TODO: We only initialize this way

        self.asked_feature = list()
        self.do_random = do_random
        self.rejected_item_list_ = list()

        self.history_list = list()

        self.write_fp = write_fp
        self.strategy = strategy
        self.TopKTaxo = TopKTaxo
        self.entropy_dict_10 = None
        self.entropy_dict_50 = None
        self.entropy_dict = None
        self.sim_dict = None
        self.sim_dict2 = None
        self.PN_model = PN_model


        self.known_feature = list()
        self.known_facet = list()

        self.residual_feature_big = None
        self.skip_big_feature = list()

        self.log_prob_list = log_prob_list
        self.action_tracker = action_tracker
        self.candidate_length_tracker = candidate_length_tracker
        self.mini_update_already = False
        self.mini = mini
        self.optimizer1_fm = optimizer1_fm
        self.optimizer2_fm = optimizer2_fm
        self.alwaysupdate = alwaysupdate
        self.previous_dict = None
        self.rejected_time = 0
        self.big_feature_length = 666
        self.feature_length = 666
        self.sample_dict = sample_dict
        self.choose_pool = choose_pool

    def get_batch_data(self, pos_neg_pairs, bs, iter_):
        PAD_IDX1 = len(cfg.user_list) + len(cfg.item_dict)
        PAD_IDX2 = cfg.feature_count

        left = iter_ * bs  # bs: batch size
        right = min((iter_ + 1) * bs, len(pos_neg_pairs))
        pos_list, pos_list2, neg_list, neg_list2 = list(), list(), list(), list()
        for instance in pos_neg_pairs[left: right]:
            # instance[0]: pos item, instance[1] neg item
            pos_list.append(torch.LongTensor([self.user_id, instance[0] + len(cfg.user_list)]))
            f = cfg.item_dict[str(instance[0])]['feature_index']  # 0.82304 TODO
            # f = [PAD_IDX2]
            pos_list2.append(torch.LongTensor(f))

            neg_list.append(torch.LongTensor([self.user_id, instance[1] + len(cfg.user_list)]))
            f = cfg.item_dict[str(instance[1])]['feature_index']
            # f = [PAD_IDX2]
            neg_list2.append(torch.LongTensor(f))
        # end for
        static_preference_index = torch.LongTensor(self.known_feature).expand(len(pos_list), len(self.known_feature))

        pos_list = pad_sequence(pos_list, batch_first=True, padding_value=PAD_IDX1)
        pos_list2 = pad_sequence(pos_list2, batch_first=True, padding_value=PAD_IDX2)

        neg_list = pad_sequence(neg_list, batch_first=True, padding_value=PAD_IDX1)
        neg_list2 = pad_sequence(neg_list2, batch_first=True, padding_value=PAD_IDX2)


        return cuda_(pos_list), cuda_(pos_list2), cuda_(neg_list), cuda_(neg_list2)

    # end def

    def mini_update(self, input_message):
        # self.FM_model.train()
        bs = 32
        r = r_list[self.history_list[-1] + 2]
        # print('bias = {}'.format(self.bias))
        # print('bias = {}'.format(self.bias))

        self.history_list = list(self.history_list)
        if self.history_list[-1] == -1 or self.history_list[-1] == 2 or input_message.message_type == 'quit':
            if self.history_list[-1] == 2:
                update_item = [input_message.data['itemID'] + len(cfg.user_list)]
            else:
                update_item = [i + len(cfg.user_list) for i in input_message.data['rejected_item_list']]

            for i in range(len(update_item)):
                item_1 = update_item[i]
                B = cfg.user_TS_matrix[self.user_id]
                b = cfg.user_emb[item_1].reshape(d, 1)
                cfg.user_TS_matrix[self.user_id] = B + np.dot(b, b.reshape([1, d]))
                cfg.user_TS_f[self.user_id] += (r - self.bias[i]) * b
                cfg.user_TS_matrix_inv[self.user_id] = np.linalg.inv(cfg.user_TS_matrix[self.user_id])
                cfg.user_emb[self.user_id] = (cfg.user_TS_matrix_inv[self.user_id].dot(cfg.user_TS_f[self.user_id])).reshape(d)
                # cfg.user_emb[self.user_id] = np.random.multivariate_normal(mean=cfg.user_emb[[self.user_id]].reshape(d), cov=v * v * cfg.user_TS_matrix_inv[self.user_id])

        if self.history_list[-1] == 1 or (self.history_list[-1] == -2 and input_message.message_type != 'quit') or self.history_list[-1] == 0:
            value = input_message.data['value']
            facet = input_message.data['facet']
            if facet == 'city':
                asked_facet = int(value[0])
            elif facet == 'stars':
                asked_facet = int(value[0] + cfg.city_count - 1)
            elif facet == 'RestaurantsPriceRange2':
                asked_facet = int(value[0] + cfg.city_count + cfg.star_count - 1)
            else:
                # base = cfg.star_count + cfg.price_count + cfg.city_count
                # asked_facet = [int(i + base) for i in value]
                asked_facet = value

            # asked_facet = list(set([asked_facet]))
            if value != None:
                if type(asked_facet) != int:
                    for i in asked_facet:
                        B = cfg.user_TS_matrix[self.user_id]
                        b = cfg.emb_matrix[i].reshape(d, 1)
                        cfg.user_TS_matrix[self.user_id] = B + np.dot(b, b.reshape([1, d]))
                        cfg.user_TS_f[self.user_id] += (r - self.bias[0]) * b
                        cfg.user_TS_matrix_inv[self.user_id] = np.linalg.inv(cfg.user_TS_matrix[self.user_id])
                        cfg.user_emb[self.user_id] = (cfg.user_TS_matrix_inv[self.user_id].dot(cfg.user_TS_f[self.user_id])).reshape(d)
                        # cfg.user_emb[self.user_id] = np.random.multivariate_normal(mean=cfg.user_emb[[self.user_id]].reshape(d),cov=v * v * cfg.user_TS_matrix_inv[self.user_id])

                else:
                    B = cfg.user_TS_matrix[self.user_id]
                    b = cfg.emb_matrix[asked_facet].reshape(d, 1)
                    cfg.user_TS_matrix[self.user_id] = B + np.dot(b, b.reshape([1, d]))
                    cfg.user_TS_f[self.user_id] += (r - self.bias[0]) * b
                    cfg.user_TS_matrix_inv[self.user_id] = np.linalg.inv(cfg.user_TS_matrix[self.user_id])
                    cfg.user_emb[self.user_id] = (cfg.user_TS_matrix_inv[self.user_id].dot(cfg.user_TS_f[self.user_id])).reshape(d)
                    # cfg.user_emb[self.user_id] = np.random.multivariate_normal(mean=cfg.user_emb[[self.user_id]].reshape(d),cov=v * v * cfg.user_TS_matrix_inv[self.user_id])

        reg_ = torch.Tensor([cfg.update_reg])
        reg_ = torch.autograd.Variable(reg_, requires_grad=False)
        reg_ = cuda_(reg_)
        reg = reg_



    def vectorize(self):
        list1 = [v for k, v in self.entropy_dict_10.items()]
        list2 = [v for k, v in self.entropy_dict_50.items()]
        list3 = [v for k, v in self.entropy_dict.items()]
        list4 = [v for k, v in self.sim_dict2.items()]

        list5 = self.history_list + [0] * (15 - len(self.history_list))

        list6 = [0] * 8
        if len(self.recent_candidate_list) <= 10:
            list6[0] = 1
        if len(self.recent_candidate_list) > 10 and len(self.recent_candidate_list) <= 50:
            list6[1] = 1
        if len(self.recent_candidate_list) > 50 and len(self.recent_candidate_list) <= 100:
            list6[2] = 1
        if len(self.recent_candidate_list) > 100 and len(self.recent_candidate_list) <= 200:
            list6[3] = 1
        if len(self.recent_candidate_list) > 200 and len(self.recent_candidate_list) <= 300:
            list6[4] = 1
        if len(self.recent_candidate_list) > 300 and len(self.recent_candidate_list) <= 500:
            list6[5] = 1
        if len(self.recent_candidate_list) > 500 and len(self.recent_candidate_list) <= 1000:
            list6[6] = 1
        if len(self.recent_candidate_list) > 1000:
            list6[7] = 1

        list_cat = list3 + list4 + list5 + list6

        list_cat = np.array(list_cat)

        assert len(list_cat) == 81
        return list_cat

    # end def

    def vectorize_crm(self):
        a = [0] * self.feature_length
        for item in self.known_feature:
            a[item] = 1
        return np.array(a)


    def cal_bias_at(self, ranked_facet):
        for index, big_feature in enumerate(cfg.FACET_POOL):
            if ranked_facet == big_feature:
                big_feature_matrix = cfg.emb_matrix[[index]]
        preference_matrix_all = np.zeros((1, d))
        preference_matrix_all = preference_matrix_all.reshape(1, d)
        given_preference = self.known_feature
        if len(given_preference) > 0:
            if len(given_preference) == 1:
                preference_matrix = cfg.emb_matrix[given_preference].reshape(1, d)
            else:
                preference_matrix = cfg.emb_matrix[given_preference]

            preference_matrix_all = preference_matrix
        bias = np.dot(preference_matrix_all, np.array(big_feature_matrix).T)
        # cosine_result = cosine_result.sum(dim=0).reshape(cosine_result.shape[0], -1)
        bias = [float(np.sum(bias, axis=1))]

        return bias


    def update_upon_feature_inform(self, input_message):
        # assert input_message.message_type == cfg.INFORM_FACET

        # _______ update F_dict________

        if input_message.message_type == cfg.INFORM_FACET:

            facet = input_message.data['facet']
            self.asked_feature += facet

            value = input_message.data['value']
            facet_lable = input_message.data['facet_lable']

            if self.history_list[-1] == 1:
                self.recent_candidate_list = [k for k in self.recent_candidate_list if
                                              set(value).issubset(set(cfg.item_dict[str(k)]['feature_index']))]
                self.recent_candidate_list = list(set(self.recent_candidate_list) - set([self.busi_id])) + [
                    self.busi_id]
                self.known_facet += facet
                fresh = True

                self.known_feature = list(set(self.known_feature))

                if fresh is True:
                    # dictionary
                    l = list(set(self.recent_candidate_list) - set([self.busi_id]))
                    random.shuffle(l)
                    if cfg.play_by == 'AOO':
                        self.sample_dict[self.busi_id].append((self.known_feature, l[: 10]))

                if cfg.play_by != 'AOO':
                    self.sim_dict = feature_similarity(self.known_feature, self.user_id, self.TopKTaxo)
                    self.sim_dict2 = self.sim_dict.copy()

            start = time.time()
            if (facet_lable != 0) or self.turn_count == 1:
                c = cal_ent(self.recent_candidate_list[: 10])
                d = c.do_job()
                self.entropy_dict_10 = d
                c = cal_ent(self.recent_candidate_list[: 50])
                d = c.do_job()
                self.entropy_dict_50 = d

                c = cal_ent(self.recent_candidate_list)
                d = c.do_job()
                self.entropy_dict = d

        # _______ update candidate db____________
        # ---------- Need to check the correctness here!-------------
        # _______ here we only generated the query sentence, will do the query after the if judgment
        start = time.time()

        self.sim_dict = feature_similarity(self.known_feature, self.user_id, self.TopKTaxo)
        self.sim_dict2 = self.sim_dict.copy()

        for f in self.asked_feature:
            if self.entropy_dict != None:
                self.entropy_dict[f] = 0

        for f in self.asked_feature:
            if self.sim_dict is not None and f in self.sim_dict:
                self.sim_dict[f] = -10
                if self.entropy_dict != None:
                    if self.entropy_dict[f] == 0:
                        self.sim_dict[f] = -10

        for f in self.asked_feature:
            if self.sim_dict2 is not None and f in self.sim_dict:
                self.sim_dict2[f] = -10
                if self.entropy_dict != None:
                    if self.entropy_dict[f] == 0:
                        self.sim_dict[f] = -10




    def prepare_next_question(self):
        if self.strategy == 'maxent':
            facet = max(self.entropy_dict, key=self.entropy_dict.get)
            data = dict()
            data['facet'] = facet
            # data['candidate'] = self.recent_candidate_list
            new_message = message(cfg.AGENT, cfg.USER, cfg.ASK_FACET, data)
            self.asked_feature.append(facet)
            return new_message
        elif self.strategy == 'maxsim':
            for f in self.asked_feature:
                if self.sim_dict is not None and f in self.sim_dict:
                    self.sim_dict[f] = -1
            # facet = max(self.sim_dict, key=self.sim_dict.get)
            if len(self.known_feature) == 0 and self.sim_dict is None:
                facet = max(self.entropy_dict, key=self.entropy_dict.get)
            else:
                current_dict = sorted(self.sim_dict.items(), key=lambda x: x[1], reverse=True)
                facet = [current_dict[i][0] for i in range(18)]

            data = dict()
            data['facet'] = facet
            # print('ask attribute = {}'.format(facet))
            # data['candidate'] = self.recent_candidate_list
            new_message = message(cfg.AGENT, cfg.USER, cfg.ASK_FACET, data)
            self.asked_feature += facet
            return new_message
        else:
            pool = [item for item in cfg.FACET_POOL if item not in self.asked_feature]
            facet = np.random.choice(np.array(pool), 1)[0]
            data = dict()
            if facet in [item.name for item in cfg.cat_tree.children]:
                data['facet'] = facet
            else:
                data['facet'] = facet
            # data['candidate'] = self.recent_candidate_list
            # print('candidate size agent: {}'.format(len(self.recent_candidate_list)))
            new_message = message(cfg.AGENT, cfg.USER, cfg.ASK_FACET, data)
            return new_message

    def prepare_rec_message(self):
        self.recent_candidate_list_ranked = [item for item in self.recent_candidate_list_ranked if
                                             item not in self.rejected_item_list_]  # Delete those has been rejected
        rec_list = self.recent_candidate_list_ranked[: 10]
        data = dict()
        data['rec_list'] = rec_list
        # print('recommend items = {}'.format(rec_list))
        new_message = message(cfg.AGENT, cfg.USER, cfg.MAKE_REC, data)

        bias = list()
        for i, item in enumerate(rec_list):
            preference_matrix_all = cfg.user_emb[[20]]
            preference_matrix_all = preference_matrix_all.reshape(1, d)
            given_preference = self.known_feature
            if len(given_preference) > 0:
                if len(given_preference) == 1:
                    preference_matrix = cfg.emb_matrix[given_preference].reshape(1, d)
                else:
                    preference_matrix = cfg.emb_matrix[given_preference]

                preference_matrix_all = np.concatenate((preference_matrix, preference_matrix_all), axis=0)
            cosine_result = float(np.sum(np.dot(preference_matrix_all, cfg.user_emb[[item + len(cfg.user_list)]].T), axis=0))
            bias.append(cosine_result)

        return new_message, bias

    def response(self, input_message):
        '''
        The agent moves a step forward, upon receiving a message from the user.
        '''
        assert input_message.sender == cfg.USER
        assert input_message.receiver == cfg.AGENT

        # _______ update the agent self_______
        if input_message.message_type == cfg.INFORM_FACET:
            if input_message.data['facet_lable'] == 0:
                self.history_list.append(0)

            else:
                self.history_list.append(1)
                cfg.ask_count[self.turn_count] += 1

            self.mini_update(input_message)

        if input_message.message_type == cfg.REJECT_REC:
            self.rejected_item_list_ += input_message.data['rejected_item_list']
            self.rejected_time += 1
            self.history_list.append(-1)
            if self.mini == 1:
                if self.alwaysupdate == 1:
                    self.mini_update(input_message)
                    self.mini_update_already = True

            # ------ update Ent ------
            c = cal_ent(self.recent_candidate_list)
            d = c.do_job()
            self.entropy_dict = d
            for f in self.asked_feature:
                self.entropy_dict[f] = 0

        self.recent_candidate_list = list(set(self.recent_candidate_list) - set(self.rejected_item_list_))
        self.recent_candidate_list = list(set(self.recent_candidate_list) - set([self.busi_id])) + [self.busi_id]

        # _______ Adding into history _______

        if input_message.message_type == cfg.ACCEPT_REC:
            self.history_list.append(2)
            self.mini_update(input_message)

        if len(self.history_list) != 0:
            if self.history_list[-1] == -2:
                self.mini_update(input_message)

        # if self.history_list[-1] != 0 or self.turn_count == 1:
        if (1 not in self.history_list) or self.history_list[-1] == 1:
            self.update_upon_feature_inform(input_message)

        self.recent_candidate_list_ranked, self.previous_dict, self.max_item_score_avg = \
            rank_items(self.known_feature, self.user_id, self.busi_id, self.skip_big_feature, self.FM_model,
                       self.recent_candidate_list, self.write_fp, 1, self.rejected_item_list_, self.previous_dict)



        action = None
        SoftMax = nn.Softmax()
        if cfg.play_by == 'AOO':
            new_message = self.prepare_next_question()  #

        if cfg.play_by == 'AO':  # means AskOnly
            action = 0
            new_message = self.prepare_next_question()

            if cfg.hardrec == 'two':
                x = len(self.recent_candidate_list)
                p = 10.0 / x
                a = random.uniform(0, 1)
                if a < p:
                    new_message = self.prepare_rec_message()

        if cfg.play_by == 'Naive':  # means AskOnly
            action = 0
            new_message = self.prepare_next_question()

            a = random.uniform(0, 1)
            if a > 0.5:
                new_message = self.prepare_rec_message()

        if cfg.play_by == 'RO':  # means RecOnly
            new_message = self.prepare_rec_message()
        if cfg.play_by == 'AR':  # means Ask and Recommend
            action = random.randint(0, 1)

        if cfg.play_by == 'policy':  # do policy gradient

            a = random.uniform(0, 1)
            print('history = {}'.format(self.history_list[-1]))
            if a < 0.8 or cfg.eval == 1:  # means choose the action with highest probability
                ranked_facet = sorted(self.sim_dict.items(), key=lambda item: item[1], reverse=True)

                if ranked_facet[0][1] > self.max_item_score_avg and (1 not in self.history_list):

                    action_max = cfg.FACET_POOL.index(ranked_facet[0][0])

                else:
                    action_max = len(cfg.FACET_POOL)
                    cfg.itm_psi[self.turn_count] += 1

                # The following line are sampled the action chosen.
                action = Variable(torch.IntTensor([action_max]))

            else:  # means we choose a random action
                action = np.random.randint(0, self.big_feature_length + 1)

            if action < len(cfg.FACET_POOL):
                data = dict()
                data['facet'] = [ranked_facet[i][0] for i in range(12)]
                new_message = message(cfg.AGENT, cfg.USER, cfg.ASK_FACET, data)
                self.bias = self.cal_bias_at(ranked_facet[0][0])
            else:
                new_message, self.bias = self.prepare_rec_message()

        if cfg.play_by == 'policy':
            self.action_tracker.append(action.data.numpy().tolist())
            self.candidate_length_tracker.append(len(self.recent_candidate_list))


        new_message.data['itemID'] = self.busi_id
        return new_message
    # end def response
# end def class agent
