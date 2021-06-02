# BB-8 and R2-D2 are best friends.

import sys
sys.path.insert(0, '../FM')
sys.path.insert(0, '../yelp')

#from FM_model import *
import itertools
import json
import pandas as pd
import pickle
import numpy as np
from collections import defaultdict
import time
import torch
from FM_old import FactorizationMachine

emb_size = 64

def cuda_(var):
    return var.cuda() if torch.cuda.is_available() else var


class _Config():
    def __init__(self):
        self.init_basic()
        self.init_type()

        self.init_misc()
        self.init_test()
        self.init_FM_related()
        self.init_bandit_parm()

    def init_basic(self):
        with open('../../data/FM-train-data/review_dict_train.json', 'r') as f:
            self._train_user_to_items = json.load(f)
        with open('../../data/FM-train-data/review_dict_valid.json', 'r') as f:
            self._valid_user_to_items = json.load(f)
        with open('../../data/FM-train-data/review_dict_test.json', 'r') as f:
            self._test_user_to_items = json.load(f)
        with open('../../data/FM-train-data/FM_busi_list.pickle', 'rb') as f:
            self.busi_list = pickle.load(f)
        with open('../../data/FM-train-data/FM_user_list.pickle', 'rb') as f:
            self.user_list = pickle.load(f)
        with open('../../data/FM-train-data/FM_train_list.pickle', 'rb') as f:
            self.train_list = pickle.load(f)
        with open('../../data/FM-train-data/FM_valid_list.pickle', 'rb') as f:
            self.valid_list = pickle.load(f)
        with open('../../data/FM-train-data/FM_test_list.pickle', 'rb') as f:
            self.test_list_2 = pickle.load(f)
        self.test_list = list()

        self.old_users = 21309

        for i in range(len(self.train_list)):
            u, item, r = self.train_list[i]
            user_id = int(u)
            if user_id >= self.old_users:
                self.test_list.append([u, item, r])

        for i in range(len(self.test_list_2)):
            u, item, r = self.test_list_2[i]
            user_id = int(u)
            if user_id >= self.old_users:
                self.test_list.append([u, item, r])

        for i in range(len(self.valid_list)):
            u, item, r = self.valid_list[i]
            user_id = int(u)
            if user_id >= self.old_users:
                self.test_list.append([u, item, r])

        # _______ String to Int _______
        with open('../../data/FM-train-data/item_map-merge.json', 'r') as f:
            self.item_map = json.load(f)
        with open('../../data/FM-train-data/user_map.json', 'r') as f:
            self.user_map = json.load(f)
        with open('../../data/FM-train-data/city_map.json', 'r') as f:
            self.city_map = json.load(f)
        with open('../../data/FM-train-data/tag_map-new.json', 'r') as f:
            self.tag_map = json.load(f)
        with open('../../data/FM-train-data/2-layer-tax-v2.json', 'r') as f:
            self.taxo_dict = json.load(f)

        self.tag_map_inverted = dict()
        for k, v in self.tag_map.items():
            self.tag_map_inverted[v] = k

        # _______ item info _______
        with open('../../data/FM-train-data/item_dict-merge.json', 'r') as f:
            self.item_dict = json.load(f)

        with open('../../data/FM-train-data/busi_name.json', 'r') as f:
            self.busi_name = json.load(f)


    def init_type(self):
        self.INFORM_FACET = 'INFORM_FACET'
        self.ACCEPT_REC = 'ACCEPT_REC'
        self.REJECT_REC = 'REJECT_REC'

        # define agent behavior
        self.ASK_FACET = 'ASK_FACET'
        self.MAKE_REC = 'MAKE_REC'
        self.FINISH_REC_ACP = 'FINISH_REC_ACP'
        self.FINISH_REC_REJ = 'FINISH_REC_REJ'
        self.EPISODE_START = 'EPISODE_START'

        # define the sender type
        self.USER = 'USER'
        self.AGENT = 'AGENT'

    def init_misc(self):
        self.FACET_POOL = ['city', 'stars', 'RestaurantsPriceRange2']
        self.FACET_POOL += self.taxo_dict.keys()
        print('Total feature length is: {}, Top 30 namely: {}'.format(len(self.FACET_POOL), self.FACET_POOL[: 30]))
        self.REC_NUM = 10
        self.MAX_TURN = 15
        self.play_by = None
        self.calculate_all = None
        self.turn_count = np.zeros((16, 1))

    def init_FM_related(self):
        city_max = 0
        category_max = 0
        feature_max = 0
        for k, v in self.item_dict.items():
            if v['city'] > city_max:
                city_max = v['city']
            if max(v['categories']) > category_max:
                category_max = max(v['categories'])
            if max(v['feature_index']) > feature_max:
                feature_max = max(v['feature_index'])

        stars_list = [1, 2, 3, 4, 5]
        price_list = [1, 2, 3, 4]
        self.star_count, self.price_count = len(stars_list), len(price_list)
        self.city_count, self.category_count, self.feature_count = city_max + 1, category_max + 1, feature_max + 1

        self.city_span = (0, self.city_count)
        self.star_span = (self.city_count, self.city_count + self.star_count)
        self.price_span = (self.city_count + self.star_count, self.city_count + self.star_count + self.price_count)

        self.spans = [self.city_span, self.star_span, self.price_span]

        print('city max: {}, category max: {}, feature max: {}'.format(self.city_count, self.category_count, self.feature_count))
        fp = '../../data/FM-model-merge/yelp-fmdata-new.pt'
        model = FactorizationMachine(emb_size=64, user_length=len(self.user_list), item_length=len(self.item_dict),
                                     feature_length=feature_max + 1, qonly=1, command=8, hs=64, ip=0.01,
                                     dr=0.5, old_new='new')
        model.load_state_dict(torch.load(fp, map_location='cpu'))
        print('load FM model {}, but hypyer parameters can be mistaken'.format(fp))
        self.emb_matrix = model.feature_emb.weight[..., :-1].detach().numpy()
        self.user_emb = model.ui_emb.weight[..., :-1].detach().numpy()
        avr_user_emb = np.zeros((1, emb_size))
        for i in range(self.old_users):
            avr_user_emb += self.user_emb[[i]].reshape(1, emb_size)
        avr_user_emb = avr_user_emb / self.old_users

        for j in range(self.old_users, len(self.user_list)):
            self.user_emb[[j]] = avr_user_emb

        self.FM_model = cuda_(model)

    def init_test(self):
        pass


    def init_bandit_parm(self):
        self.user_TS_matrix = [np.eye(emb_size, emb_size) for i in range(len(self.user_emb))]
        self.user_TS_matrix_inv = [np.eye(emb_size, emb_size) for i in range(len(self.user_emb))]
        self.user_TS_f = [self.user_emb[i].reshape(emb_size, 1) for i in range(len(self.user_emb))]



    def change_param(self, playby, eval, update_count, update_reg, purpose, mod):
        self.play_by = playby
        self.eval = eval
        self.update_count = update_count
        self.update_reg = update_reg
        self.purpose = purpose
        self.mod = mod

        if self.mod == 'crm':
           category_max = 0
           feature_max = 0
           for k, v in self.item_dict.items():
               if max(v['categories']) > category_max:
                   category_max = max(v['categories'])
               if max(v['feature_index']) > feature_max:
                   feature_max = max(v['feature_index'])
           fp = '../../data/FM-model-merge/v32-test-FM-lr-0.01-flr-0.0001-reg-0.002-decay-0.0-qonly-1-bs-64-command-6-hs-64-ip-0.01-dr-0.5-optim-Ada-oldnew-new-pretrain-0-uf-0-rd-0-freeze-0-seed-3702-useremb-1epoch-45.pt'
           fp = '../../data/FM-model-merge/v32-test-FM-lr-0.01-flr-0.0001-reg-0.002-decay-0.0-qonly-1-bs-64-command-6-hs-64-ip-0.01-dr-0.5-optim-Ada-oldnew-new-pretrain-0-uf-0-rd-0-freeze-0-seed-3704-useremb-1epoch-49.pt'

           fp = '../../data/FM-model-merge/v32-test-FM-lr-0.01-flr-0.001-reg-0.005-decay-0.0-qonly-1-bs-64-command-8-hs-64-ip-0.01-dr-0.5-optim-Ada-oldnew-new-pretrain-2-uf-1-rd-0-freeze-0-seed-3702-useremb-1epoch-5.pt'

           model = FactorizationMachine(emb_size=64, user_length=len(self.user_list), item_length=len(self.item_dict),
                                        feature_length=feature_max + 1, qonly=1, command=6, hs=64, ip=0.01,
                                        dr=0.5, old_new='new')
           start = time.time()
           model.load_state_dict(torch.load(fp))
           print('load FM model {} takes: {} secs, but hypyer parameters can be mistaken'.format(fp,
                                                                                                 time.time() - start))
           #self.emb_matrix = model.feature_emb.weight[..., :-1].detach().numpy()
           #self.user_emb = model.ui_emb.weight[..., :-1].detach().numpy()
           self.FM_model = cuda_(model)


start = time.time()
global_config = _Config()
print('Config takes: {}'.format(time.time() - start))

print('___Config Done!!___')
