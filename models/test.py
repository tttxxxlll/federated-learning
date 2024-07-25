#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def test_img(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss = 0
    banchloss = []
    data_loader = DataLoader(datatest, batch_size=args.bs, shuffle=True)
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.cuda(), target.cuda()
        log_probs = net_g(data)
        log_probs = log_probs.to(torch.float32)
        target = target.to(torch.float32)
        # sum up batch loss

        test_loss = F.mse_loss(log_probs, target).item()
        banchloss.append(test_loss)
        # get the index of the max log-probability


    loss_avg = sum(banchloss)/len(banchloss)

    # if args.verbose:
    #     print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
    #         test_loss, correct, len(data_loader.dataset), accuracy))
    return  loss_avg

