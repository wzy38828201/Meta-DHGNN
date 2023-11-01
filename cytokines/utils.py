import datetime
import dgl
import errno
import numpy as np
import os
import pickle
import random
import torch
import pandas as pd

from dgl.data.utils import download, get_download_dir, _get_dgl_url
from pprint import pprint
from scipy import sparse
from scipy import io as sio

from get_adj import get_undirected_adj,get_pr_directed_adj,get_appr_directed_adj,get_second_directed_adj
from 邻接矩阵 import ad
from 特征矩阵 import fea

def set_random_seed(seed=0):
    """Set random seed.
    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def mkdir_p(path, log=True):
    """Create a directory for the specified path.
    Parameters
    ----------
    path : str
        Path name
    log : bool
        Whether to print result for directory creation
    """
    try:
        os.makedirs(path)
        if log:
            print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and log:
            print('Directory {} already exists.'.format(path))
        else:
            raise

def get_date_postfix():
    """Get a date based postfix for directory name.
    Returns
    -------
    post_fix : str
    """
    dt = datetime.datetime.now()
    post_fix = '{}_{:02d}-{:02d}-{:02d}'.format(
        dt.date(), dt.hour, dt.minute, dt.second)

    return post_fix

def setup_log_dir(args, sampling=False):
    """Name and create directory for logging.
    Parameters
    ----------
    args : dict
        Configuration
    Returns
    -------
    log_dir : str
        Path for logging directory
    sampling : bool
        Whether we are using sampling based training
    """
    date_postfix = get_date_postfix()
    log_dir = os.path.join(
        args['log_dir'],
        '{}_{}'.format(args['dataset'], date_postfix))

    if sampling:
        log_dir = log_dir + '_sampling'

    mkdir_p(log_dir)
    return log_dir

# The configuration below is from the paper.
default_configure = {
    'lr': 0.005,#0.005             # Learning rate
    'lr2': 0.001,#0.01
    'num_heads': [8],#8        # Number of attention heads for node-level attention
    'hidden_units': 8,#8
    'dropout': 0.6,#0.6
    'weight_decay': 0.003,#0.001
    'weight_decay2': 0.0001,#0.001
    'num_epochs': 20,#200
    'num_epochs1': 162,#72
    'patience': 100#100
}

sampling_configure = {
    'batch_size': 20
}

def setup(args):
    args.update(default_configure)
    set_random_seed(args['seed'])
    args['dataset'] = 'ACMRaw' if args['hetero'] else 'ACM'
    args['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    args['log_dir'] = setup_log_dir(args)
    return args

def setup_for_sampling(args):
    args.update(default_configure)
    args.update(sampling_configure)
    set_random_seed()
    args['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    args['log_dir'] = setup_log_dir(args, sampling=True)
    return args

def get_binary_mask(total_size, indices):
    mask = torch.zeros(total_size)
    mask[indices] = 1
    return mask.byte()

def load_acm(remove_self_loop):
    # url = 'dataset/ACM3025.pkl'
    # data_path = get_download_dir() + '/ACM3025.pkl'
    # download(_get_dgl_url(url), path=data_path)
    #
    # with open(data_path, 'rb') as f:
    #     data = pickle.load(f)

    # labels, features = torch.from_numpy(data['label'].todense()).long(), \
    #                    torch.from_numpy(data['feature'].todense()).float()
    # num_classes = labels.shape[1]
    # labels = labels.nonzero()[:, 1]
    truefeatures = np.array(np.identity(400))
    truefeatures = torch.Tensor(np.array(truefeatures).tolist())
    features = truefeatures
    # if remove_self_loop:
    #     num_nodes = data['label'].shape[0]
    #     data['PAP'] = sparse.csr_matrix(data['PAP'] - np.eye(num_nodes))
    #     data['PLP'] = sparse.csr_matrix(data['PLP'] - np.eye(num_nodes))

    # Adjacency matrices for meta path based neighbors
    # (Mufei): I verified both of them are binary adjacency matrices with self loops

    # author_g = dgl.from_scipy(data['PAP'])
    # subject_g = dgl.from_scipy(data['PLP'])

    #矩阵有向的变换
    M_ = ad()
    adj_q = sparse.csr_matrix(M_)

    coo = adj_q.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    indices = torch.from_numpy(indices).long()
    # labels = torch.from_numpy(labels).long()
    edge_index, edge_weight, L1 = get_appr_directed_adj(0.1, indices, features.shape[0], features.dtype)
    # data = Data(x=features, edge_index=edge_index, edge_weight=edge_weight, y=labels)
    edge_index, edge_weight, L2 = get_second_directed_adj(indices, features.shape[0], features.dtype)
    L0 = np.identity(len(L1))
    L0 = sparse.csr_matrix(L0)
    L1 = np.array(L1)
    L1 = sparse.csr_matrix(L1)
    L2 = np.array(L2)
    L2 = sparse.csr_matrix(L2)

    author_g = dgl.from_scipy(L1)
    subject_g = dgl.from_scipy(L2)
    L0_g = dgl.from_scipy(L0)
    gs = [author_g, subject_g]
    r_g = L0_g
    # #不用有向的变换
    # r_g = dgl.from_scipy(adj_q)
    # r_g = dgl.add_self_loop(r_g)#添加自循环解决0度节点问题
    # #gs = [author_g, subject_g]
    # gs = [r_g]

    www1 = pd.read_excel('label.xlsx', index=None)
    tr1 = www1[www1[2] == 1].index.tolist()#正标签
    tr2 = www1[www1[3] == 1].index.tolist()#负标签
    tr1_ = tr1[:int(len(tr1)*0.2)]#正标签的训练部分
    tr2_ = tr2[int(len(tr2)*0.8):]#负标签的训练部分
    tr1__ = [n1 for n1 in tr1 if n1 not in tr1_]#正标签的测试部分
    tr2__ = [n1 for n1 in tr2 if n1 not in tr2_]#负标签的测试部分
    tr1___ = tr1__[:int(len(tr1__)*0.5)]#正标签的验证部分
    tr2___ = tr2__[:int(len(tr2__) * 0.5)]#负标签的验证部分
    tr1____ = [n2 for n2 in tr1__ if n2 not in tr1___]#正标签的测试部分
    tr2____ = [n2 for n2 in tr2__ if n2 not in tr2___]#负标签的测试部分

    tr3 = np.array([tr1_ + tr2_])#训练集
    tr3_ = np.array([tr1___ + tr2___])#验证集
    tr3__ = np.array([tr1____ + tr2____])#测试集
    labels = www1[4]
    # for yu in range(len(www1)):
    #     if yu in tr1:
    #         labels.append(int(0))
    #     elif yu in tr2:
    #         labels.append(int(1))
    #     else:
    #         labels.append(int(2))
    # labels = np.array(labels)
    labels = torch.tensor(labels)

    train_idx = torch.from_numpy(tr3).long().squeeze(0)
    val_idx = torch.from_numpy(tr3_).long().squeeze(0)
    test_idx = torch.from_numpy(tr3__).long().squeeze(0)

    num_nodes = r_g.number_of_nodes()#author_g，r_g
    train_mask = get_binary_mask(num_nodes, train_idx)
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)

    print('dataset loaded')
    pprint({
        'dataset': '细胞因子',
        'train': train_mask.sum().item() / num_nodes,
        'val': val_mask.sum().item() / num_nodes,
        'test': test_mask.sum().item() / num_nodes
    })

    return gs, features, labels, train_idx, val_idx, test_idx, train_mask, val_mask, test_mask

def load_acm_raw(remove_self_loop):
    assert not remove_self_loop
    url = 'dataset/ACM.mat'
    data_path = get_download_dir() + '/ACM.mat'
    download(_get_dgl_url(url), path=data_path)

    data = sio.loadmat(data_path)
    p_vs_l = data['PvsL']       # paper-field?
    p_vs_a = data['PvsA']       # paper-author
    p_vs_t = data['PvsT']       # paper-term, bag of words
    p_vs_c = data['PvsC']       # paper-conference, labels come from that

    # We assign
    # (1) KDD papers as class 0 (data mining),
    # (2) SIGMOD and VLDB papers as class 1 (database),
    # (3) SIGCOMM and MOBICOMM papers as class 2 (communication)
    conf_ids = [0, 1, 9, 10, 13]
    label_ids = [0, 1, 2, 2, 1]

    p_vs_c_filter = p_vs_c[:, conf_ids]
    p_selected = (p_vs_c_filter.sum(1) != 0).A1.nonzero()[0]
    p_vs_l = p_vs_l[p_selected]
    p_vs_a = p_vs_a[p_selected]
    p_vs_t = p_vs_t[p_selected]
    p_vs_c = p_vs_c[p_selected]

    hg = dgl.heterograph({
        ('paper', 'pa', 'author'): p_vs_a.nonzero(),
        ('author', 'ap', 'paper'): p_vs_a.transpose().nonzero(),
        ('paper', 'pf', 'field'): p_vs_l.nonzero(),
        ('field', 'fp', 'paper'): p_vs_l.transpose().nonzero()
    })

    features = torch.FloatTensor(p_vs_t.toarray())

    pc_p, pc_c = p_vs_c.nonzero()
    labels = np.zeros(len(p_selected), dtype=np.int64)
    for conf_id, label_id in zip(conf_ids, label_ids):
        labels[pc_p[pc_c == conf_id]] = label_id
    labels = torch.LongTensor(labels)

    num_classes = 3

    float_mask = np.zeros(len(pc_p))
    for conf_id in conf_ids:
        pc_c_mask = (pc_c == conf_id)
        float_mask[pc_c_mask] = np.random.permutation(np.linspace(0, 1, pc_c_mask.sum()))
    train_idx = np.where(float_mask <= 0.2)[0]
    val_idx = np.where((float_mask > 0.2) & (float_mask <= 0.3))[0]
    test_idx = np.where(float_mask > 0.3)[0]

    num_nodes = hg.number_of_nodes('paper')
    train_mask = get_binary_mask(num_nodes, train_idx)
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)

    return hg, features, labels, num_classes, train_idx, val_idx, test_idx, \
            train_mask, val_mask, test_mask

def load_data(dataset, remove_self_loop=False):
    if dataset == 'ACM':
        return load_acm(remove_self_loop)
    elif dataset == 'ACMRaw':
        return load_acm_raw(remove_self_loop)
    else:
        return NotImplementedError('Unsupported dataset {}'.format(dataset))

class EarlyStopping(object):
    def __init__(self, patience=10):
        dt = datetime.datetime.now()
        self.filename = 'early_stop_{}_{:02d}-{:02d}-{:02d}.pth'.format(
            dt.date(), dt.hour, dt.minute, dt.second)
        self.patience = patience
        self.counter = 0
        self.best_acc = None
        self.best_loss = None
        self.early_stop = False

    def step(self, loss, acc, model):
        if self.best_loss is None:
            self.best_acc = acc
            self.best_loss = loss
            self.save_checkpoint(model)
        elif (loss > self.best_loss) and (acc < self.best_acc):
            self.counter += 1
            print('EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if (loss <= self.best_loss) and (acc >= self.best_acc):
                self.save_checkpoint(model)
            self.best_loss = np.min((loss, self.best_loss))
            self.best_acc = np.max((acc, self.best_acc))
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        torch.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model):
        """Load the latest checkpoint."""
        model.load_state_dict(torch.load(self.filename))
