#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms

def iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items=[1300, 2200, 200, 3200, 200, 400, 700, 100, 200, 100, 700, 500, 3000, 400, 200, 1700, 500, 1200, 300, 1600, 100, 300, 700, 700, 300, 1000, 200, 200, 100, 300, 200, 700, 300, 1500, 400, 1200, 2700, 100, 100, 200]
    # for i in range(len(num_items)):
    #     num_items[i]=int(num_items[i]*0.8)
    # num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items[i], replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
        # print(len(dict_users[i]))
    return dict_users,num_items


