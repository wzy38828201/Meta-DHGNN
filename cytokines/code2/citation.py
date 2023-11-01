import torch
import torch.nn.functional as F
import torch.optim as optim
from itertools import combinations
import random

from correspondingsequencenumbername import xuhao_mingcheng
from Labelsorting import biaoqianfenlei

import scipy.stats as stats
from args import get_citation_args
from utils import load_citation, set_seed
from models import get_model
from metrics import accuracy,accuracy1
import pandas as pd
import os
import shutil
import numpy as np

pd.set_option('display.max_columns', 100000000)
pd.set_option('display.width', 100000000)
pd.set_option('display.max_colwidth', 100000000)
pd.set_option('display.max_rows', 100000000)


def train_regression(model, train_features, train_labels, idx_train, epochs, weight_decay, lr, adj):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(epochs):#只在训练里面有epoch
        model.train()
        optimizer.zero_grad()#把梯度置零，也就是把loss关于weight的导数变成0,对于每个batch大都执行了这样的操作
        output = model(train_features, adj)
        loss_train = F.cross_entropy(output[idx_train], train_labels[idx_train])
        #print('Step:', epoch, '\tMeta_Training_Loss:', loss_train)
        loss_train.backward()#调用backward()函数之前都要将梯度清零，反向传播求解梯度
        optimizer.step()#更新权重参数
    return model, loss_train, optimizer

def train_regression1(model, train_features, train_labels, idx_train, epochs1, weight_decay, lr, adj):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(epochs1):#只在训练里面有epoch
        model.train()
        optimizer.zero_grad()#把梯度置零，也就是把loss关于weight的导数变成0,对于每个batch大都执行了这样的操作
        output = model(train_features, adj)
        loss_train = F.cross_entropy(output[idx_train], train_labels[idx_train])
        #print('Step:', epoch, '\tMeta_Training_Loss:', loss_train)
        loss_train.backward()#调用backward()函数之前都要将梯度清零，反向传播求解梯度
        optimizer.step()#更新权重参数
    return model, loss_train, optimizer

#Train Model

def test_regression(model, test_features, test_labels, idx_test, adj):
    model.eval()
    output = model(test_features, adj)
    loss_train = F.cross_entropy(output[idx_test], test_labels[idx_test])
    acc_ = accuracy(output[idx_test], test_labels[idx_test])
    return acc_, loss_train

def test_regression1(model, test_features, test_labels, idx_test, adj):
    model.eval()
    output = model(test_features, adj)
    pred_q = F.softmax(output, dim=1).detach().tolist()
    acc_ = accuracy1(output[idx_test], test_labels[idx_test])

    #print(acc_)
    return pred_q
#Test Model

def reset_array():
    class1_train = []
    class2_train = []
    class1_test = []
    class2_test = []
    train_idx = []
    test_idx = []
#Clear Array


def main():
    args = get_citation_args()          #get args
    n_way = args.n_way                  #how many classes
    train_shot_0 = args.train_shot_0        #train-shot
    train_shot_1 = args.train_shot_1
    test_shot_0 = args.test_shot_0          #test-shot
    test_shot_1 = args.test_shot_1
    step = args.step
    step1 = args.step1
    node_num = args.node_num
    iteration = args.iteration

    accuracy_meta_test = []
    total_accuracy_meta_test = []

    set_seed(args.seed, args.cuda)
    #set seed
    #adj, features, labels = load_citation(args.dataset, args.normalization, args.cuda)
    #load dataset

    # if args.dataset == 'cora':
    #     class_label = [0, 1, 2, 3, 4, 5, 6]
    #     combination = list(combinations(class_label, n_way))
    # elif args.dataset == 'citeseer':
    #     node_num = 3327
    #     iteration = 15
    #     class_label = [0, 1, 2, 3, 4, 5]
    #     combination = list(combinations(class_label, n_way))
    if args.dataset == 'PPI':
        node_num = 1615
        class_label = [0, 1]
        combination = list(combinations(class_label, 2))

    wai = 0
    aa = pd.read_csv(open(r'G:\元学习\Meta-GNN-master\gcn_fewshot\1769\标签分类\序号和名称的对应.csv'), header=None, index_col=False)
    dic1 = {}
    for kk in aa[1]:
        dic1[kk] = 0
    dic2 = {}
    for kk1 in aa[1]:
        dic2[kk1] = 0

    list1 = []
    for mmh in range(1000):
        path_0 = r'G:\元学习\Meta-GNN-master\gcn_fewshot\1769\1000\save' + str(wai)
        if os.path.exists(path_0):
            shutil.rmtree(path_0)
        os.makedirs(path_0)
        #随机选取test_shot_0个负标签
        with open(r'G:\元学习\Meta-GNN-master\gcn_fewshot\1769\随机筛选.txt','r') as f:
            ss = f.readlines()
            jieguo = random.sample(ss,test_shot_0)
        #随机选取的test_shot_0个负标签写入
        with open(r'G:\元学习\Meta-GNN-master\gcn_fewshot\1769\label负.txt', 'w') as f:
            for iii in jieguo:
                f.write(iii)

        shu = 0
        mmm = 0

        #for mmm in range(10):
        while mmm==0:
            xuhao_mingcheng()
            biaoqianfenlei()
            adj, features, labels = load_citation(args.dataset, args.normalization, args.cuda)
            #for i in range(len(combination)):
            #print('a')
            #print('Cross_Validation: ',i+1)
            test_label = list(combination[0])
            train_label = [n for n in class_label if n not in test_label]
            # print('Cross_Validation {} Train_Label_List {}: '.format(i+1, train_label))
            # print('Cross_Validation {} Test_Label_List {}: '.format(i+1, test_label))
            model = get_model(args.model, features.size(1), n_way, args.hidden, args.dropout, args.cuda).cuda()
            #create model
            path = path_0+'\\'+str(shu)
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path)

            train_loss0 = []
            train_loss1 = []
            val_loss = []

            for j in range(iteration):
                #print('a')
                labels_local = labels.clone().detach()
                #select_class = random.sample(train_label, n_way)
                select_class = [0, 1]
                # print('EPOCH {} ITERATION {} Train_Label: {}'.format(i+1, j+1, select_class))
                class1_idx = []
                class2_idx = []

                for k in range(node_num):
                    if(labels_local[k] == select_class[0]):
                        class1_idx.append(k)
                        #labels_local[k] = 0
                    elif(labels_local[k] == select_class[1]):
                        class2_idx.append(k)
                        #labels_local[k] = 1

                for m in range(step):
                    class1_train = random.sample(class1_idx,train_shot_0)
                    class2_train = random.sample(class2_idx,train_shot_1)
                    class1_test = [n1 for n1 in class1_idx if n1 not in class1_train]
                    class2_test = [n2 for n2 in class2_idx if n2 not in class2_train]
                    train_idx = class1_train + class2_train
                    random.shuffle(train_idx)
                    test_idx = class1_test + class2_test
                    random.shuffle(test_idx)

                    model,loss_train0,optimizer= train_regression(model, features, labels_local, train_idx, args.epochs, args.weight_decay, args.lr, adj)
                    train_loss0.append(loss_train0.item())
                    acc_query,loss_val = test_regression(model, features, labels_local, test_idx, adj)#验证机的准确率
                    val_loss.append(loss_val.item())
                    reset_array()
                    #if loss_val.item()<0.05:
                    print('Step:', j, '\tMeta_Training_Loss:', loss_val.item())

            pd.DataFrame(val_loss).to_csv(path+'\\'+'val_loss.csv',header=None)
            pd.DataFrame(train_loss0).to_csv(path+'\\'+'train_loss0.csv', header=None)
            torch.save(model.state_dict(), path+'\\'+'model.pkl')
            #save model as 'model.pkl'

            labels_local = labels.clone().detach()
            select_class = select_class = [0, 1]
            #print('EPOCH {} Test_Label {}: '.format(i+1, select_class))
            class1_idx1 = []
            class2_idx1 = []
            reset_array()
            s = []
            lb = list(range(node_num))

            for k in range(node_num):
                if(labels_local[k] == select_class[0]):
                    class1_idx1.append(k)
                    #labels_local[k] = 0
                elif(labels_local[k] == select_class[1]):
                    class2_idx1.append(k)
                    #labels_local[k] = 1
            class_idx = class1_idx1 + class2_idx1
            class_idx_test = np.sort(class_idx)
            for m in lb:
                if m not in class_idx_test:
                    s.append(m)  # 不包括取的0和1标签的测试集

            #
            for m in range(step1):
                class1_train = random.sample(class1_idx1, test_shot_0-2)
                class2_train = random.sample(class2_idx1, test_shot_1)
                # class1_test = [n1 for n1 in class1_idx if n1 not in class1_train]
                # class2_test = [n2 for n2 in class2_idx if n2 not in class2_train]
                train_idx = class1_train + class2_train
                random.shuffle(train_idx)
                #test_idx = class1_test + class2_test
                test_idx = s
                random.shuffle(test_idx)

                model_meta_trained = get_model(args.model, features.size(1), n_way, args.hidden, args.dropout, args.cuda).cuda()
                model_meta_trained.load_state_dict(torch.load(path+'\\'+'model.pkl'))
                #re-load model 'model.pkl'

                model_meta_trained,loss_train1,optimizer = train_regression1(model_meta_trained, features, labels_local, train_idx, args.epochs1, args.weight_decay, args.lr, adj)
                train_loss1.append(loss_train1.item())
                pred_q  = test_regression1(model_meta_trained, features, labels_local, test_idx, adj)#测试集的准确率
                print('Step:', j, '\tMeta_Training_Loss:', loss_train1.item())
                reset_array()

            pd.DataFrame(train_loss1).to_csv(path+'\\'+'train_loss1.csv', header=None)
            pd.DataFrame(pred_q).to_csv(path+'\\'+'pred_q.csv', header=None)
            label_num = pd.read_csv(open(r'G:\元学习\Meta-GNN-master\gcn_fewshot\1769\标签分类\细胞因子名称、编码、序号.csv'), header=None,index_col=False)
            j = label_num.loc[lb]
            j.columns = ['1','2','3']
            j.sort_values('2', inplace=True, ascending=True)
            #pd.DataFrame(j).to_csv(path+'\\'+'结果对应的序号和名称.csv', header=None)
            jg = pd.read_csv(open(path+'\\'+'pred_q.csv'), header=None,index_col=False)
            jg = jg.iloc[:,2]

            df = pd.concat([j, jg], axis=1)
            df.columns = ['1','2','3','4']
            df.sort_values('4', inplace=True,ascending=False)
            df.index = range(len(j))
            pd.DataFrame(df).to_csv(path + '\\' + '结果对应的序号和名称.csv', header=None)

            #取最后100个细胞因子的编码
            dd = df['1'].iloc[len(j)-100-test_shot_0:len(j)-100].tolist()
            with open(r'G:\元学习\Meta-GNN-master\gcn_fewshot\1769\label负.txt', 'w') as f:
                for iii in dd:
                    f.write(iii+'\n')

            # for hhh in val_loss:
            #     if hhh<0.02:
            #         print(hhh)
            #         mmm=1
            # if shu!=0:
            #     list1 = []
            #     list2 = []
            #     a = pd.read_csv(open(path + '\\' + '结果对应的序号和名称.csv'),header=None, index_col=False)
            #     a_a = a[1].iloc[len(a)-100:len(a)].tolist()
            #     pathl = path_0+'\\'+str(shu-1)
            #     b = pd.read_csv(open(pathl + '\\' + '结果对应的序号和名称.csv'), header=None, index_col=False)
            #     b_b = b[1].iloc[len(b)-100:len(b)].tolist()
            #     a_b = a_a==b_b
            #     if a_b==True:
            #         mmm = 1
            a = pd.read_csv(open(path + '\\' + '结果对应的序号和名称.csv'), header=None, index_col=False)
            if shu==1:
                mmm=1
            print(mmm)
            list1.append(shu)
            shu = shu+1

        #将最后一次的结果秩的和进行输入
        for ooo, ppp in enumerate(a[3]):
            dic1[ppp] = dic1[ppp] + ooo
        for hhh in val_loss:
            if hhh < 0.02:
                for ooo1, ppp1 in enumerate(a[3]):
                    dic2[ppp1] = dic2[ppp1] + ooo1
                break


        wai = wai+1

    #print(list1)
    pd.DataFrame.from_dict(dic1,orient='index').to_csv(r'C:\Users\lenovo\Desktop\结果(全部).csv', header=None)
    pd.DataFrame.from_dict(dic2,orient='index').to_csv(r'C:\Users\lenovo\Desktop\结果(收敛).csv', header=None)

if __name__ == '__main__':
    main()
