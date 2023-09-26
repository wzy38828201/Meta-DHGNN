import numpy as np
import scipy.sparse as sp
import torch
import sys
import pickle as pkl
import networkx as nx
from normalization import fetch_normalization, row_normalize
from sklearn.metrics import f1_score
import pandas as pd

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
    #names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    #objects = []
    # for i in range(len(names)):
    #     with open("data/ind.{}.{}".format(dataset_str.lower(), names[i]), 'rb') as f:
    #         if sys.version_info > (3, 0):#查看sys的版本的，是否大于3.0
    #             objects.append(pkl.load(f, encoding='latin1'))
    #         else:
    #             objects.append(pkl.load(f))

    #x, y, tx, ty, allx, ally, graph = tuple(objects)
    #test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    #pd.DataFrame(test_idx_reorder).to_csv(r'G:\元学习\Meta-GNN-master\meta_gnn\自己的数据\运行的结果\test_idx_reorder.csv')
    #test_idx_range = np.sort(test_idx_reorder)
    #print(test_idx_range)
    # if dataset_str == 'citeseer':
    #     test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
    #     tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
    #     tx_extended[test_idx_range-min(test_idx_range), :] = tx
    #     tx = tx_extended
    #     ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
    #     ty_extended[test_idx_range-min(test_idx_range), :] = ty
    #     ty = ty_extended

    #features = sp.vstack((allx, tx)).tolil()
    #features[test_idx_reorder, :] = features[test_idx_range, :]
    #adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    #print(adj)
    #adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #print(adj)

    #计算adj的
    adjlist_path = open(r'G:\元学习\Meta-GNN-master\gcn_fewshot\1769\全部的细胞因子.tsv')
    adjlist_file = pd.read_table(adjlist_path, sep="\t")
    adjlist = adjlist_file[["node1_external_id", "node2_external_id", "combined_score"]]
    #
    # all_protein = adjlist["node1_external_id"]
    # all_protein = pd.concat([adjlist["node1_external_id"], adjlist["node2_external_id"]])
    # all_protein = all_protein.drop_duplicates()
    # protein_map = pd.DataFrame(np.arange(len(all_protein)), index=all_protein, columns=["nodes"])
    # protein_map.to_csv(r'G:\元学习\Meta-GNN-master\meta_gnn\全部的细胞因子预测\全部的细胞因子节点映射.csv')

    protein_map = pd.read_csv(open(r'G:\元学习\Meta-GNN-master\gcn_fewshot\1769\全部的细胞因子节点映射.csv'), index_col=0)
    adjlist_node1 = protein_map.loc[adjlist["node1_external_id"]]
    adjlist_node2 = protein_map.loc[adjlist["node2_external_id"]]
    adjlist.loc[:, "node1_external_id"] = adjlist_node1["nodes"].values
    adjlist.loc[:, "node2_external_id"] = adjlist_node2["nodes"].values
    adjlist = adjlist.rename(columns={'node1_external_id': 'node1', 'node2_external_id': 'node2'})
    # adjlist['combined_score'] = adjlist['combined_score'] / 1000
    # 用01来去判断的，也可以用小数来去判断，方便灵活使用的(通过设定not weighted的值来去决定是否使用这个函数)
    adj_matrix1 = sp.csr_matrix((adjlist['combined_score'], (adjlist['node1'], adjlist['node2'])),
                                shape=(len(protein_map), len(protein_map)))
    adj_matrix2 = sp.triu(adj_matrix1) + sp.tril(adj_matrix1).T

    data = []
    row = []
    col = []
    for i in range(len(protein_map)):
        row .append(i)# 行指标
    for i in range(len(protein_map)):
        col.append(i)# 列指标
    for i in range(len(protein_map)):
        data.append(1) # 在行指标列指标下的数字
    adj_unit = sp.csr_matrix((data, (row, col)), shape=(len(protein_map), len(protein_map)))

    adj = adj_unit + adj_matrix2 + adj_matrix2.T

    #labels = np.vstack((ally, ty))
    #print(labels)
    #labels[test_idx_reorder, :] = labels[test_idx_range, :]
    #print(labels)
    #pd.DataFrame(labels).to_csv(r'G:\元学习\Meta-GNN-master\meta_gnn\自己的数据\运行的结果\labels0.csv')

    #特征的单位矩阵
    features = sp.identity(len(protein_map))

    adj, features = preprocess_citation(adj, features, normalization)
    #print(adj)
    features = torch.FloatTensor(np.array(features.todense())).float()
    #labels = torch.LongTensor(labels)
    #print(labels)
    a = pd.read_csv(r'G:\元学习\Meta-GNN-master\gcn_fewshot\1769\标签分类\全部的因子分类.csv', index_col=0)
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
