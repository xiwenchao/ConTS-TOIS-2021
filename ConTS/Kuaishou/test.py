from config_3 import global_config as cfg
import numpy as np
import json

with open('../../data/FM-train-data/item_dict-merge.json', 'r') as f:
    item_dict = json.load(f)

k = 0
n = 0
for key in item_dict:
    k += 1
    n += len(item_dict[key]['feature_index'])

print('average_tag = {}'.format(n / k))

