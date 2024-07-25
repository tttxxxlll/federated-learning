from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import copy
import torch
import math
from utils.options import args_parser
import copy
import numpy as np

args = args_parser()


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg
# w=np.arange(20).reshape(5,4)
# print(w)

def defense(w):
    l = 5
    a=0
    H=0
    diffh=[0 for index in range(int(args.num_users*args.frac))]
    sortdiffh = [0 for index in range(int(args.num_users*args.frac))]
    diffh1 = [0 for index in range(int(args.num_users*args.frac)-1  )]
    attcilent=[]
    for i in range(500):
        h = [0 for index in range(500)]
        p = [0,0,0,0,0]
        m = [0, 0, 0, 0, 0]
        d=(np.max(w[i])-np.min(w[i]))/l
        for e in range(l):
            for j in range(int(args.num_users*args.frac)):
                if w[i,j]>=np.min(w[i])+(d*e) and w[i,j]<=np.min(w[i])+(d*(e+1)):
                     m[e]=m[e]+1
        for n in range(l):
            p[n]=m[n]/(args.num_users*args.frac)
            if p[n]!=0:
                h[i] =p[n] * (1 - p[n])
        H+=h[i]
    hk = [0 for index in range(int(args.num_users*args.frac))]
    for g in range(int(args.num_users*args.frac)):
        HK=0
        for i in range(500):
            h = [0 for index in range(500)]
            p = [0, 0, 0, 0, 0]
            m = [0, 0, 0, 0, 0]
            d = (np.max(w[i]) - np.min(w[i])) / l
            for e in range(l):
                for j in range(int(args.num_users*args.frac)):
                    if j!=g:
                        if w[i, j] >= np.min(w[i]) + (d * e) and w[i, j] <= np.min(w[i]) + (d * (e + 1)):
                            m[e] = m[e] + 1
            for n in range(l):
                p[n] = m[n] / 20
                if p[n] != 0:
                    h[i] = p[n] * (1 - p[n])
            HK+= h[i]
        hk[g]=HK
    for i in range(int(args.num_users*args.frac)):
        diffh[i]=H-hk[i]
    sortdiffh=sorted(diffh,reverse=True)
    for i in range(int(args.num_users*args.frac)-3):
        diffh1[i]=abs((sortdiffh[i+1]-sortdiffh[i]))
    # print(sortdiffh)
    # print(diffh1)
    if np.max(diffh1)<=1:
        a=0
        attcilent = []
    else:
        a=np.argmax(diffh1)+1

        for i in range(a):
            for j in range(int(args.num_users*args.frac)):
                if sortdiffh[i]==diffh[j]:
                    attcilent.append(j)
                    diffh[j]=0
        attcilent.sort()

    return a,attcilent

def krum(w):
    distances = defaultdict(dict)
    non_malicious_count = 12
    num = 0
    for k in w[0].keys():
        if num == 0:
            for i in range(len(w)):
                for j in range(i):
                    distances[i][j] = distances[j][i] = np.linalg.norm(w[i][k].cpu().numpy() - w[j][k].cpu().numpy())
            num = 1
        else:
            for i in range(len(w)):
                for j in range(i):
                    distances[j][i] += np.linalg.norm(w[i][k].cpu().numpy() - w[j][k].cpu().numpy())
                    distances[i][j] += distances[j][i]
    minimal_error = 1e20
    for user in distances.keys():
        errors = sorted(distances[user].values())
        current_error = sum(errors[:non_malicious_count])
        if current_error < minimal_error:
            minimal_error = current_error
            minimal_error_index = user
    return w[minimal_error_index]

def trimmed_mean(w,args):
    number_to_consider = 12
    print(number_to_consider)
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        tmp = []
        for i in range(len(w)):
            tmp.append(w[i][k].cpu().numpy()) # get the weight of k-layer which in each client
        tmp = np.array(tmp)
        med = np.median(tmp,axis=0)
        new_tmp = []
        for i in range(len(tmp)):# cal each client weights - median
            new_tmp.append(tmp[i]-med)
        new_tmp = np.array(new_tmp)
        good_vals = np.argsort(abs(new_tmp),axis=0)[:number_to_consider]
        good_vals = np.take_along_axis(new_tmp, good_vals, axis=0)
        k_weight = np.array(np.mean(good_vals) + med)
        w_avg[k] = torch.from_numpy(k_weight).to(args.device)
    return w_avg





