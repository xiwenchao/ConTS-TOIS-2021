# BB-8 and R2-D2 are best friends.

import sys
sys.path.insert(0, '../FM')
sys.path.insert(0, '../yelp')

import pickle
import torch
import argparse

import time
import numpy as np
import json

from config_3 import global_config as cfg
from FM_old import FactorizationMachine
from epi_3 import run_one_episode, update_PN_model
# from pn import PolicyNetwork
from FM_old import FactorizationMachine
import copy

from collections import defaultdict

import random
import os.path
import json
import math
import regex as re

random.seed(1)

the_max = 0
for k, v in cfg.item_dict.items():
    if the_max < max(v['feature_index']):
        the_max = max(v['feature_index'])
print(the_max)
FEATURE_COUNT = the_max + 1


def cuda_(var):
    return var.cuda() if torch.cuda.is_available()else var


def main():
    parser = argparse.ArgumentParser(description="Run conversational recommendation.")
    parser.add_argument('-mt', type=int, dest='mt', help='MAX_TURN')
    parser.add_argument('-playby', type=str, dest='playby', help='playby')
    parser.add_argument('-fmCommand', type=str, dest='fmCommand', help='fmCommand')
    parser.add_argument('-optim', type=str, dest='optim', help='optimizer')
    parser.add_argument('-lr', type=float, dest='lr', help='lr')
    parser.add_argument('-decay', type=float, dest='decay', help='decay')
    parser.add_argument('-TopKTaxo', type=int, dest='TopKTaxo', help='TopKTaxo')
    parser.add_argument('-gamma', type=float, dest='gamma', help='gamma')
    parser.add_argument('-trick', type=int, dest='trick', help='trick')

    parser.add_argument('-startFrom', type=int, dest='startFrom', help='startFrom')
    parser.add_argument('-endAt', type=int, dest='endAt', help='endAt')
    parser.add_argument('-strategy', type=str, dest='strategy', help='strategy')
    parser.add_argument('-eval', type=int, dest='eval', help='eval')
    parser.add_argument('-mini', type=int, dest='mini', help='mini')  # means mini-batch update the FM
    parser.add_argument('-alwaysupdate', type=int, dest='alwaysupdate',
                        help='alwaysupdate')  # means mini-batch update the FM
    parser.add_argument('-initeval', type=int, dest='initeval', help='initeval')
    parser.add_argument('-upoptim', type=str, dest='upoptim', help='upoptim')
    parser.add_argument('-uplr', type=float, dest='uplr', help='uplr')
    parser.add_argument('-upcount', type=int, dest='upcount', help='upcount')
    parser.add_argument('-upreg', type=float, dest='upreg', help='upreg')
    parser.add_argument('-code', type=float, dest='code', help='code')
    parser.add_argument('-purpose', type=str, dest='purpose', help='purpose')  # options: pretrain, fmdata, others
    parser.add_argument('-mod', type=str, dest='mod', help='mod')  # options: CRM, EAR

    A = parser.parse_args()


    cfg.change_param(playby=A.playby, eval=A.eval, update_count=A.upcount, update_reg=A.upreg,
                     purpose=A.purpose, mod=A.mod)

    chat_history_list = list()

    #TODO: uncomment this to undo shuffle
    random.seed(1)
    random.shuffle(cfg.valid_list)
    random.shuffle(cfg.test_list)

    the_valid_list = copy.copy(cfg.valid_list)
    the_test_list = copy.copy(cfg.test_list)
    random.shuffle(the_valid_list)
    random.shuffle(the_test_list)

    print('valid length: {}, test list length: {}'.format(len(the_valid_list), len(the_test_list)))
    #sys.sleep(1)


    gamma = A.gamma
    FM_model = cfg.FM_model
    #PN_model = cfg.PN_model

    INPUT_DIM = 0
    if A.mod == 'ear':
        INPUT_DIM = 81
    if A.mod == 'crm':
        INPUT_DIM = 590

    # PN_model = PolicyNetwork(input_dim=INPUT_DIM, dim1=64, output_dim=34, r_or_t=A.rt)  # for baselines with policy network
    PN_model = 'PolicyNetwork'
    start = time.time()

    start_point = A.startFrom
    end_point = A.endAt

    sample_dict = defaultdict(list)

    total_turn = 0
    for epi_count in range(A.startFrom, A.endAt): # A.endAt
        if epi_count % 100 == 0:
            print('It has processed {} episodes'.format(epi_count))
        if epi_count >= len(cfg.test_list):
            continue
        start = time.time()

        # TODO: It is very important, to copy the model
        # Following for initialize FM model for each episode
        current_FM_model = copy.deepcopy(FM_model)
        cuda_(current_FM_model)
        param1, param2 = list(), list()
        param3 = list()
        i = 0
        for name, param in current_FM_model.named_parameters():
            # print(name, param)
            if i == 0:
                param1.append(param)
            else:
                param2.append(param)
            if i == 2:
                param3.append(param)
                param.requires_grad = False
            i += 1
        #optimizer1_fm = torch.optim.Adagrad(param1, lr=0.01, weight_decay=A.decay)
        #optimizer2_fm = torch.optim.SGD(param2, lr=0.001, weight_decay=A.decay)

        # following old code
        optimizer1_fm, optimizer2_fm = None, None
        if A.purpose != 'fmdata':
            optimizer1_fm = torch.optim.Adagrad(param1, lr=0.01, weight_decay=A.decay)  # TODO: change learning rate
            if A.upoptim == 'Ada':
                optimizer2_fm = torch.optim.Adagrad(param2, lr=A.uplr, weight_decay=A.decay)
            if A.upoptim == 'SGD':
                optimizer2_fm = torch.optim.SGD(param2, lr=0.001, weight_decay=A.decay)
        # end following


        if A.purpose != 'fmdata':
            u, item = cfg.test_list[epi_count]
            user_id = int(u)
            item_id = int(item)
        else:
            user_id = 0
            item_id = epi_count

        big_feature_list = list()

        print('\n\n\nHello! I am glad to serve you!')

        write_fp = '../../data/interaction-log/{}/v6-code-{}-s-{}-e-{}-lr-{}-gamma-{}-playby-{}-stra-{}-topK-{}-trick-{}-eval-{}-init-{}-mini-{}-always-{}-upcount-{}-upreg-{}-m.txt'.format(
            A.mod.lower(), A.code, A.startFrom, A.endAt, A.lr, A.gamma, A.playby, A.strategy, A.TopKTaxo, A.trick,
            A.eval, A.initeval,
            A.mini, A.alwaysupdate, A.upcount, A.upreg)

        #print('Starting new\nuser ID: {}, item ID: {}, episode count: {}\n'.format(user_id, item_id, epi_count))

        choose_pool = cfg.item_dict[str(item_id)]['feature_index'][:3]

        choose_pool_original = choose_pool
        if A.purpose not in ['pretrain', 'fmdata']:
            choose_pool = [random.choice(choose_pool)]


        # run the episode

        for c in choose_pool:
            start_facet = c
            if A.purpose != 'pretrain':
                log_prob_list, rewards, turn_count = run_one_episode(current_FM_model, user_id, item_id, A.mt, False,
                                                                     write_fp,
                                                                     A.strategy, A.TopKTaxo,
                                                                     PN_model, gamma, A.trick,
                                                                     A.mini,
                                                                     optimizer1_fm, optimizer2_fm, A.alwaysupdate,
                                                                     start_facet, sample_dict,
                                                                     choose_pool_original)
            else:
                current_np = run_one_episode(current_FM_model, user_id, item_id, A.mt, False, write_fp,
                                             A.strategy, A.TopKTaxo,
                                             PN_model, gamma, A.trick, A.mini,
                                             optimizer1_fm, optimizer2_fm, A.alwaysupdate, start_facet,
                                             sample_dict,
                                             choose_pool_original)

            # end run
            total_turn += turn_count
            # update PN model
            if A.playby == 'policy' and A.eval != 1:
                # if rewards is None:
                #    print('?')
                # update_PN_model(PN_model, log_prob_list, rewards, optimizer)
                None
            # end update



    print('average_turn = {}'.format(total_turn / (- A.startFrom + A.endAt)))
    print('successful recommend rate = {}'.format(cfg.turn_count / (- A.startFrom + A.endAt)))



if __name__ == '__main__':
    main()