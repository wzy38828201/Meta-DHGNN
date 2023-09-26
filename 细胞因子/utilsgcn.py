import numpy as np
import scipy.sparse as sp
import torch
import sys
import pickle as pkl
import networkx as nx
from normalization import fetch_normalization, row_normalize
from sklearn.metrics import f1_score
import pandas as pd
from scipy.sparse import coo_matrix

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def f1(output, labels):
    preds = output.max(1)[1]
    preds = preds.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    micro = f1_score(labels, preds, average='micro')
    return micro


def set_seed(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def preprocess_citation(adj, features, normalization="FirstOrderGCN"):
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    features = row_normalize(features)
    return adj, features


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_citation(dataset_str='cora', normalization='FirstOrderGCN', cuda=True):
    a = pd.read_excel(r'G:\图神经网络编程\有向异构\dglhan改\实验\string_interactions.xlsx', header=None)
    c = pd.read_excel(r'G:\图神经网络编程\有向异构\dglhan改\实验\细胞因子名称.xlsx', header=None)
    a1 = a[0].tolist()
    a2 = a[1].tolist()
    a3 = a1 + a2
    a4 = a[2].tolist()
    b = pd.Series(a3).unique().tolist()
    # d = c[3].tolist()
    # e = [i for i in d if i not in b]
    # for i,j in enumerate():
    d = dict(zip(c[1], c[0]))
    f = []
    f1 = []
    for i in a1:
        if i in d.keys():
            f.append(d[i])
    for ii in a2:
        if ii in d.keys():
            f1.append(d[ii])
    a_ = pd.DataFrame()
    a_[0] = f
    a_[1] = f1
    a_[2] = a4
    # 构造邻接矩阵
    adj = coo_matrix((a_.iloc[:, 2], (a_.iloc[:, 0], a_.iloc[:, 1])), shape=(len(c), len(c)))

    #labels = np.vstack((ally, ty))
    #print(labels)
    #labels[test_idx_reorder, :] = labels[test_idx_range, :]
    #print(labels)
    #pd.DataFrame(labels).to_csv(r'G:\元学习\Meta-GNN-master\meta_gnn\自己的数据\运行的结果\labels0.csv')

    #特征的单位矩阵
    features = sp.identity(400)

    adj, features = preprocess_citation(adj, features, normalization)

    #print(adj)
    features = torch.FloatTensor(np.array(features.todense())).float()
    #labels = torch.LongTensor(labels)
    #print(labels)
    a = pd.read_csv(r'G:\图神经网络编程\有向异构\dglhan改\实验\全部的因子分类.csv', index_col=0)
    b = a.iloc[:, 0].values
    b[:] = b[:]
    labels = torch.LongTensor(b)
    #print(labels)
    #labels = torch.max(labels, dim=1)[1]#有关labels的这些程序只是用来提取labels的
    #pd.DataFrame(labels).to_csv(r'G:\元学习\Meta-GNN-master\meta_gnn\自己的数据\运行的结果\labels1.csv')
    #print(labels)
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    #print(adj)

    if cuda:
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()

    return adj, features, labels
