
import matplotlib

import zijiandata

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import math
import random
from utils.sampling import iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import CNNzijian
from models.Fed import FedAvg, defense, krum, trimmed_mean
from models.test import test_img
from matplotlib import pyplot as plt

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    dataset_train = zijiandata.train_dataset
    dataset_test = zijiandata.test_dataset

    dict_users,num_data = iid(dataset_train, args.num_users)
    print(num_data)
    img_size = dataset_train[0][0].shape
    print(img_size)

    # build model
    net_glob = CNNzijian()
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    test_acc,test_loss=[],[]

    if args.all_clients:
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
    users=list(range(args.num_users))


    TF=[]
    TE=[]
    TT=[]
    TD=[]
    for iter in range(args.epochs):
        print("第",iter,"轮")
        l=[10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]#车辆与路边单元的距离：10-30米
        ll = np.random.choice(l, 20, replace=True)  # 随机选择距离

        h=[]#信道增益
        e=[]#二次调整轮数
        last_param = []
        last_param1 = []
        loss_locals = []
        num_data1=[]
        localep=[]
        localep1=[]
        efl=[]#fedavg计算能耗
        efc=[]#fedavg通信能耗
        ef=[]#fedavg总能耗
        eel=[]#我们的计算能耗
        eec=[]#我们的通信能耗
        ee=[]#我们的总能耗
        tf=[]#fedavg时间
        te=[]#我们的时间
        tt=[]#并行通信时间
        td=[]#减少通信时间
        dd=[]
        user=0
        if not args.all_clients:
            w_locals = []
            w_locals1 = []
        m=20
        if iter>=1:
            for j in range(len(att)):
                users.remove(att[j])
        print(users)
        idxs_users = np.random.choice(users, m, replace=False)
        idxs_users.sort()

        print('随机选取用户集合：',idxs_users)
        for i in idxs_users:
            num_data1.append(num_data[i])
        print("用户数据集长度：",num_data1)
        for i in range(20):#初次制定
            localep.append(sum(num_data1)/num_data1[i])
        print("初次制定轮数:",localep)
        ll[np.argmax(num_data1)]=28
        print("用户距离:", ll)
        for i in range(20):
            h.append(math.pow(10, -1.78) * math.pow(ll[i], -2.2))
        print("用户信道增益：",h)
        print((max(h)+min(h))/2)

        for i in range(20):
            dd.append( (0.3*210549)/(1e7*np.log2(1+((0.3*h[i])/(1e-3)))))
        for i in range(20):
            a=(max(dd)+min(dd))/2
            b = (0.3*210549)/(1e7*np.log2(1+((0.3*h[i])/(1e-3))))
            e.append((a-b)/(1e-28*num_data1[i]*3e5*1.5*1e9*1.5*1e9))
        print("二次调整轮数增量：", e)
        for i in range(20):
            localep1.append(int(localep[i]+e[i]))
        for i in range(20):
            if localep1[i]<=0:
                localep1[i]=1
        print("二次调整后轮数：", localep1)
        for i in range(20):
            efl.append(20*1e-28*num_data1[i]*3e5*1.5*1e9*1.5*1e9)
            efc.append((0.3*210549)/(1e7*np.log2(1+((0.3*h[i])/(1e-3)))))
            eel.append(localep1[i]*1e-28*num_data1[i]*3e5*1.5*1e9*1.5*1e9)
            eec.append((0.3*210549)/(1e7*np.log2(1+((0.3*h[i])/(1e-3)))))
        for i in range(20):
            ef.append(efl[i]+efc[i])
            ee.append(eel[i]+eec[i])
        print("fedavg能耗",ef)
        print("我们的能耗",ee)
        for i in range(20):
            tf.append(num_data1[i]*20*3e5/(1.5*1e9) + (210549)/(1e7*np.log2(1+((0.3*h[i])/(1e-3)))))
            te.append(num_data1[i]*localep1[i]*3e5/(1.5*1e9) + (210549)/(1e7*np.log2(1+((0.3*h[i])/(1e-3)))))
            tt.append(num_data1[i] * 20 * 3e5 / (1.5 * 1e9) )
            td.append(num_data1[i] *2 * 20 * 3e5 / (1.5 * 1e9) + (210549) / (1e7 * np.log2(1 + ((0.3 * h[i]) / (1e-3)))))
        print("fedavg时间", max(tf))
        print("我们的时间", max(te))
        print( max(tt))
        print( max(td))
        TF.append(max(tf))
        TE.append(max(te))
        TT.append(max(tt))
        TD.append(max(td))
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device),local_ep=localep1[user])
            user+=1

            if idx ==2 or idx ==4 or idx ==5 or idx ==6 or idx ==7 or idx ==11 or idx ==14 or idx ==16  or idx ==18 or idx ==21 or idx ==22 or idx ==24 or idx ==26 or idx ==28 or idx ==30 or idx ==32:

                 for name in w:
                     # w[name].data +=torch.normal(0, 4, w[name].data.shape)
                    w[name].data = w[name].data*-5###符号翻转攻击

            last_param.append(torch.flatten(w['fc2.weight'].data).tolist())

            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        last_param1 = np.matrix(last_param).T

        Attclient=[]
        att=[]
        A,Attclient=defense(last_param1)
        if A!=0 :
            for i in range(len(Attclient)):
                att.append(idxs_users[Attclient[i]])
            print('共检测出%d个恶意用户' %A)
            print("恶意用户集合：", att)
            w_locals1= [i for num, i in enumerate(w_locals) if num not in Attclient]
            w_glob = FedAvg(w_locals1)
        else:
            print('共检测出%d个恶意用户' % A)
            w_glob = FedAvg(w_locals)


        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.6f}'.format(iter, loss_avg))

        loss_train.append(loss_avg)
        net_glob.eval()
        loss_test = test_img(net_glob, dataset_test, args)

        print("Testing loss: {:.6f}".format(loss_test))

        test_loss.append(loss_test)

    print('test loss:', test_loss)
    print('train loss:', loss_train)
    print("fedavg每轮用时：",TF)
    print("我们的每轮用时：", TE)
    print("并行通信每轮用时：", TT)
    print("减少通信每轮用时：", TD)

