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
from Adjacency_matrix import ad
from Characteristic_matrix import fea

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

## The configuration below is from the paper.
#default_configure = {
#    'lr': 0.005,#0.005             # Learning rate
#    'lr2': 0.005,
#    'num_heads': [8],#8        # Number of attention heads for node-level attention
#    'hidden_units': 8,#8
#    'dropout': 0.6,#0.6
#    'weight_decay': 0.001,#0.001
#    'weight_decay2': 0.001,#0.001
#    'num_epochs': 20,#200
#    'num_epochs1': 100,#200
#    'patience': 100#100
#}

sampling_configure = {
    'batch_size': 20
}

def setup(args):
    default_configure = {
        'lr': args['lr'],  # 0.005             # Learning rate
        'lr2': args['lr2'],
        'num_heads': args['num_heads'],  # 8        # Number of attention heads for node-level attention
        'hidden_units': args['hidden_units'],  # 8
        'dropout': args['dropout'],  # 0.6
        'weight_decay': args['weight_decay'],  # 0.001
        'weight_decay2': args['weight_decay2'],  # 0.001
        'num_epochs': args['num_epochs'],  # 200
        'num_epochs1': args['num_epochs1'],  # 200
        'patience': args['patience']  # 100
    }
    print(default_configure)
    args.update(default_configure)
    set_random_seed(args['seed'])
    #args['dataset'] = 'ACMRaw' if args['hetero'] else 'ACM'
    args['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    args['log_dir'] = setup_log_dir(args)
    return args

def setup_for_sampling(args):
    default_configure = {
        'lr': args['lr'],  # 0.005             # Learning rate
        'lr2': args['lr2'],
        'num_heads': args['num_heads'],  # 8        # Number of attention heads for node-level attention
        'hidden_units': args['hidden_units'],  # 8
        'dropout': args['dropout'],  # 0.6
        'weight_decay': args['weight_decay'],  # 0.001
        'weight_decay2': args['weight_decay2'],  # 0.001
        'num_epochs': args['num_epochs'],  # 200
        'num_epochs1': args['num_epochs1'],  # 200
        'patience': args['patience']  # 100
    }
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
#    url = 'dataset/ACM3025.pkl'
#    data_path = get_download_dir() + '/ACM3025.pkl'
#    download(_get_dgl_url(url), path=data_path)
#
#    with open(data_path, 'rb') as f:
#        data = pickle.load(f)

    # labels, features = torch.from_numpy(data['label'].todense()).long(), \
    #                    torch.from_numpy(data['feature'].todense()).float()
    # num_classes = labels.shape[1]
    # labels = labels.nonzero()[:, 1]


    # Fe = fea()
    # pd.DataFrame(Fe).to_excel('\cytokines\Characteristic_matrix.xlsx', index=None, header=None)
    #tff = np.eye(1734)
    tff = pd.read_excel('\cytokines\experiments\Characteristic_matrix.xlsx', header=None)
    truefeatures = np.array(tff)
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
    from scipy.sparse import coo_matrix

    c = pd.read_excel('\cytokines\experiments\CGC.xlsx',header = None)
    M_ = coo_matrix(c).toarray()

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

    c11 = pd.read_excel('\cytokines\experiments\CPC.xlsx',header = None)
    M_1 = coo_matrix(c11).toarray()
    adj_q1 = sparse.csr_matrix(M_1)
    coo1 = adj_q1.tocoo()
    values = coo1.data
    indices1 = np.vstack((coo1.row, coo1.col))
    indices1 = torch.from_numpy(indices1).long()
    # labels = torch.from_numpy(labels).long()
    edge_index, edge_weight, L1_ = get_appr_directed_adj(0.1, indices1, features.shape[0], features.dtype)
    # data = Data(x=features, edge_index=edge_index, edge_weight=edge_weight, y=labels)
    edge_index, edge_weight, L2_ = get_second_directed_adj(indices1, features.shape[0], features.dtype)
    L0_ = np.identity(len(L1_))
    L0_ = sparse.csr_matrix(L0_)
    L1_ = np.array(L1_)
    L1_ = sparse.csr_matrix(L1_)
    L2_ = np.array(L2_)
    L2_ = sparse.csr_matrix(L2_)
    author_g_ = dgl.from_scipy(L1_)
    subject_g_ = dgl.from_scipy(L2_)
    L0_g_ = dgl.from_scipy(L0_)

    author_g2 = dgl.from_scipy(L0+L1+L2)
    subject_g2 = dgl.from_scipy(L0_+L1_+L2_)
    # L0_g2 = dgl.from_scipy(L0+L0_)
    # author_g2 = dgl.from_scipy(L1+L2)
    # subject_g2 = dgl.from_scipy(L1_ + L2_)

    gs = [author_g2, subject_g2]#, author_g_, subject_g_]
    #gs = [L0_g, author_g, subject_g , L0_g_, author_g_, subject_g_]

    www1 = pd.read_excel('\cytokines\experiments\label_f.xlsx', index=None)
    tr10 = www1[www1[2] == 1].index.tolist()
    tr1 = tr10[int(0.2*len(tr10)):]#正标签训练集
    tr1_ = www1[www1[2] == 0].index.tolist()
    tr20 = www1[www1[3] == 1].index.tolist()
    tr2 = tr20[int(0.2*len(tr10)):]#负标签训练集
    ty0 = www1[www1[2] == 1].index.tolist()
    ty = ty0[:int(0.2*len(tr10))]#正标签验证集
    ty_0 = www1[www1[3] == 1].index.tolist()
    ty_ = ty_0[:int(0.2*len(tr10))]#负标签验证集

    #tr2_ = www1[www1[4] == 1].index.tolist()
    tr3 = np.array([tr1 + ty+tr2+ty_])#训练集
    zh = np.array([tr1 + ty])#正标签
    fu = np.array([tr2 + ty_])#负标签

    ty3 = np.array([ty + ty_])#验证集
    tr3_ = np.array([tr1 + tr1_+ty])#测试集
    listt = tr3_.tolist()
    labels = []
    for yu in range(len(www1)):
        if yu in zh:
            labels.append(int(0))
        elif yu in fu:
            labels.append(int(1))
        else:
            labels.append(int(2))
    # labels = np.array(labels)
    labels = torch.tensor(labels)

    train_idx = torch.from_numpy(tr3).long().squeeze(0)
    val_idx = torch.from_numpy(ty3).long().squeeze(0)
    test_idx = torch.from_numpy(tr3_).long().squeeze(0)

    num_nodes = author_g.number_of_nodes()
    train_mask = get_binary_mask(num_nodes, train_idx)
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)

    print('dataset loaded')
    pprint({
        'dataset': 'cytokines',
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
    if dataset == 'CYTO':
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
