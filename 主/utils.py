import datetime
import dgl
import errno
import numpy as np
import os
import pickle
import random
import torch

from dgl.data.utils import download, get_download_dir, _get_dgl_url
from pprint import pprint
from scipy import sparse
from scipy import io as sio

from get_adj import get_undirected_adj,get_pr_directed_adj,get_appr_directed_adj,get_second_directed_adj

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
    # args['dataset'] = 'ACMRaw' if args['hetero'] else 'ACM'
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
    # url = 'dataset/ACM3025.pkl'
    # data_path = get_download_dir() + '/ACM3025.pkl'
    # download(_get_dgl_url(url), path=data_path)
    data_path = 'data/ACM3025.pkl'

    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    ds = data['label']
    labels, features = torch.from_numpy(data['label'].todense()).long(), \
                       torch.from_numpy(data['feature'].todense()).float()
    num_classes = labels.shape[1]
    labels = labels.nonzero()[:, 1]

    if remove_self_loop:
        num_nodes = data['label'].shape[0]
        data['PAP'] = sparse.csr_matrix(data['PAP'] - np.eye(num_nodes))
        data['PLP'] = sparse.csr_matrix(data['PLP'] - np.eye(num_nodes))

    #Adjacency matrices for meta path based neighbors
    #(Mufei): I verified both of them are binary adjacency matrices with self loops
    author_g = dgl.from_scipy(data['PAP'])
    subject_g = dgl.from_scipy(data['PLP'])
    gs = [author_g,subject_g]

    # yu = data['PAP']
    # #矩阵有向的变换
    # coo = data['PAP'].tocoo()
    # values = coo.data
    # indices = np.vstack((coo.row, coo.col))
    # indices = torch.from_numpy(indices).long()
    # # labels = torch.from_numpy(labels).long()
    # edge_index, edge_weight, L1 = get_appr_directed_adj(0.1, indices, features.shape[0], features.dtype)
    # # data = Data(x=features, edge_index=edge_index, edge_weight=edge_weight, y=labels)
    # edge_index, edge_weight, L2 = get_second_directed_adj(indices, features.shape[0], features.dtype)
    # L0 = np.identity(len(L1))
    # L0 = sparse.csr_matrix(L0)
    # L1 = np.array(L1)
    # L1 = sparse.csr_matrix(L1)
    # L2 = np.array(L2)
    # L2 = sparse.csr_matrix(L2)
    # author_g = dgl.from_scipy(L1)
    # subject_g = dgl.from_scipy(L2)
    # L0_g = dgl.from_scipy(L0)
    #
    # coo1 = data['PLP'].tocoo()
    # values = coo1.data
    # indices1 = np.vstack((coo1.row, coo1.col))
    # indices1 = torch.from_numpy(indices1).long()
    # # labels = torch.from_numpy(labels).long()
    # edge_index, edge_weight, L1_ = get_appr_directed_adj(0.1, indices1, features.shape[0], features.dtype)
    # # data = Data(x=features, edge_index=edge_index, edge_weight=edge_weight, y=labels)
    # edge_index, edge_weight, L2_ = get_second_directed_adj(indices1, features.shape[0], features.dtype)
    # L0_ = np.identity(len(L1_))
    # L0_ = sparse.csr_matrix(L0_)
    # L1_ = np.array(L1_)
    # L1_ = sparse.csr_matrix(L1_)
    # L2_ = np.array(L2_)
    # L2_ = sparse.csr_matrix(L2_)
    # author_g_ = dgl.from_scipy(L1_)
    # subject_g_ = dgl.from_scipy(L2_)
    # L0_g_ = dgl.from_scipy(L0_)
    # L0_g2 = dgl.from_scipy(L0+L0_)
    #
    # author_g2 = dgl.from_scipy(L0+L1+L2)
    # subject_g2 = dgl.from_scipy(L0_+L1_+L2_)
    #
    # gs = [author_g2 , subject_g2]#, author_g_, subject_g_]
    # #gs = [L0_g, author_g, subject_g, author_g_, subject_g_]

    train_idx = torch.from_numpy(data['train_idx']).long().squeeze(0)
    val_idx = torch.from_numpy(data['val_idx']).long().squeeze(0)
    test_idx = torch.from_numpy(data['test_idx']).long().squeeze(0)

    for t in data['train_idx']:
        tr = t
    for i in data['test_idx']:
        length = len(i)
        random.shuffle(i)
        arr = i.tolist()[:int(length*0.0)]
    new_tr = tr.tolist()+arr
    tra = []
    tra.append(new_tr)
    data_train = np.array(tra)
    train_idx = torch.from_numpy(data_train).long().squeeze(0)

    num_nodes = author_g.number_of_nodes()
    train_mask = get_binary_mask(num_nodes, train_idx)
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)

    print('dataset loaded')
    pprint({
        'dataset': 'ACM',
        'train': train_mask.sum().item() / num_nodes,
        'val': val_mask.sum().item() / num_nodes,
        'test': test_mask.sum().item() / num_nodes
    })

    return gs, features, labels, num_classes, train_idx, val_idx, test_idx, \
           train_mask, val_mask, test_mask

def load_dblp(path, remove_self_loop):
    # {APA, APCPA, APTPA}
    data = sio.loadmat(path)
    # print(data)

    labels, features = torch.from_numpy(data['label']).long(), \
                       torch.from_numpy(data['features'].astype(float)).float()
    num_classes = labels.shape[1] # 4
    labels = labels.nonzero()[:, 1]
    if remove_self_loop:
        num_nodes = features.shape[0]
        data['net_APTPA'] = sparse.csr_matrix(data['net_APTPA'] - np.eye(num_nodes))
        data['net_APCPA'] = sparse.csr_matrix(data['net_APCPA'] - np.eye(num_nodes))
        data['net_APA'] = sparse.csr_matrix(data['net_APA'] - np.eye(num_nodes))

    # 论文原始的gs构建
    # APTPA_g = dgl.from_scipy(sparse.csr_matrix(data['net_APTPA']))
    # APCPA_g = dgl.from_scipy(sparse.csr_matrix(data['net_APCPA']))
    # APA_g = dgl.from_scipy(sparse.csr_matrix(data['net_APA']))
    # gs = [APTPA_g, APCPA_g, APA_g]


    # 矩阵有向的变换
    coo = sparse.csr_matrix(data['net_APTPA']).tocoo()
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
    APTPA_g = dgl.from_scipy(L1)
    APTPA_g2 = dgl.from_scipy(L0 + L1+L2)

    coo1 = sparse.csr_matrix(data['net_APCPA']).tocoo()
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
    APCPA_g2 = dgl.from_scipy(L0_ + L1_ + L2_)

    coo11 = sparse.csr_matrix(data['net_APA']).tocoo()
    values = coo11.data
    indices11 = np.vstack((coo11.row, coo11.col))
    indices11 = torch.from_numpy(indices11).long()
    # labels = torch.from_numpy(labels).long()
    edge_index, edge_weight, L1__ = get_appr_directed_adj(0.1, indices11, features.shape[0], features.dtype)
    # data = Data(x=features, edge_index=edge_index, edge_weight=edge_weight, y=labels)
    edge_index, edge_weight, L2__ = get_second_directed_adj(indices11, features.shape[0], features.dtype)
    L0__ = np.identity(len(L1__))
    L0__ = sparse.csr_matrix(L0__)
    L1__ = np.array(L1__)
    L1__ = sparse.csr_matrix(L1__)
    L2__ = np.array(L2__)
    L2__ = sparse.csr_matrix(L2__)
    author_g__ = dgl.from_scipy(L1__)
    subject_g__ = dgl.from_scipy(L2__)
    L0_g__ = dgl.from_scipy(L0__)

    APA_g2 = dgl.from_scipy(L0__ + L1__ + L2__)

    gs = [APTPA_g2, APCPA_g2, APA_g2]

    train_idx = torch.from_numpy(data['train_idx']).long().squeeze(0)
    val_idx = torch.from_numpy(data['val_idx']).long().squeeze(0)
    test_idx = torch.from_numpy(data['test_idx']).long().squeeze(0)

    for t in data['train_idx']:
        tr = t
    for i in data['test_idx']:
        length = len(i)
        random.shuffle(i)
        arr = i.tolist()[:int(length*0.03)]
    new_tr = tr.tolist()+arr
    tra = []
    tra.append(new_tr)
    data_train = np.array(tra)
    train_idx = torch.from_numpy(data_train).long().squeeze(0)

    num_nodes = APTPA_g.number_of_nodes()
    train_mask = get_binary_mask(labels.shape[0], train_idx)
    val_mask = get_binary_mask(labels.shape[0], val_idx)
    test_mask = get_binary_mask(labels.shape[0], test_idx)

    print('DBLP dataset loaded')
    pprint({
        'dataset': 'DBLP',
        'train': train_mask.sum().item() / num_nodes,
        'val': val_mask.sum().item() / num_nodes,
        'test': test_mask.sum().item() / num_nodes
    })

    return gs, features, labels, num_classes, train_idx, val_idx, test_idx, \
           train_mask, val_mask, test_mask

def load_imdb(path,remove_self_loop):
    MAM_path = path + 'MAM.csv'
    MDM_path = path + 'MDM.csv'
    label = path + 'label.csv'
    feature = path + 'feature.csv'
    test_id = path + 'test_idx.csv'
    train_id = path + 'train_idx.csv'
    val_id = path + 'val_idx.csv'
    import pandas as pd
    data_MAM_path = pd.read_csv(MAM_path, header=None)
    data_MDM_path = pd.read_csv(MDM_path, header=None)
    data_label = pd.read_csv(label, header=None)
    data_feature = pd.read_csv(feature, header=None)
    data_test_id = pd.read_csv(test_id, header=None)
    data_train_id = pd.read_csv(train_id, header=None)
    data_val_id = pd.read_csv(val_id, header=None)
    ar_data_MAM_path = np.array(data_MAM_path)
    ar_data_MDM_path = np.array(data_MDM_path)
    ar_data_label = np.array(data_label)
    ar_data_feature = np.array(data_feature)
    ar_data_test_id = np.array(data_test_id)
    ar_data_train_id = np.array(data_train_id)
    ar_data_val_id = np.array(data_val_id)

    labels1, features = torch.from_numpy(ar_data_label).long(), \
                       torch.from_numpy(ar_data_feature.astype(float)).float()
    num_classes = labels1.shape[1]
    labels = labels1.nonzero()[:, 1]

    y = labels.tolist()
    # y_ = []
    # for tt in y:
    #     if int(tt)==0:
    #         y_.append(3)
    #     else:
    #         y_.append(tt)
    # y_ = torch.from_numpy(np.array(pd.Series(y_))).long()
    # labels = y_

    if remove_self_loop:
        num_nodes = features.shape[0]
        ar_data_MAM_path = sparse.csr_matrix(ar_data_MAM_path - np.eye(num_nodes))
        ar_data_MDM_path = sparse.csr_matrix(ar_data_MDM_path - np.eye(num_nodes))

    # # 论文原始的gs构建
    # author_g = dgl.from_scipy(sparse.csr_matrix(ar_data_MAM_path))
    # subject_g = dgl.from_scipy(sparse.csr_matrix(ar_data_MDM_path))
    # gs = [author_g,subject_g]
    # 矩阵有向的变换
    coo = sparse.csr_matrix(ar_data_MAM_path).tocoo()
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

    coo1 = sparse.csr_matrix(ar_data_MDM_path).tocoo()
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
    L0_g2 = dgl.from_scipy(L0 + L0_)
    author_g2 = dgl.from_scipy(L0 + L1+L2)
    subject_g2 = dgl.from_scipy(L0_ + L1_ + L2_)

    gs = [author_g2, subject_g2]#, author_g_, subject_g_]
    # gs = [L0_g, author_g, subject_g , L0_g_, author_g_, subject_g_]

    train_idx = torch.from_numpy(ar_data_train_id).long().squeeze(0)
    val_idx = torch.from_numpy(ar_data_val_id).long().squeeze(0)
    test_idx = torch.from_numpy(ar_data_test_id).long().squeeze(0)

    tr = ar_data_train_id
    trrr = ar_data_test_id
    length = len(trrr)
    trrrr = random.sample(list(trrr),length)
    arr = trrr.tolist()[:int(length*0.04)]
    new_tr = tr.tolist()+arr
    tra = []
    tra.append(new_tr)
    data_train = np.array(tra)
    train_idx = torch.from_numpy(data_train).long().squeeze(0)

    num_nodes = author_g.number_of_nodes()
    train_mask = get_binary_mask(labels.shape[0], train_idx)
    val_mask = get_binary_mask(labels.shape[0], val_idx)
    test_mask = get_binary_mask(labels.shape[0], test_idx)

    print('dataset loaded')
    print('train',train_mask.sum().item() / num_nodes)
    print('val', val_mask.sum().item() / num_nodes)
    print('test', test_mask.sum().item() / num_nodes)
    print({
        'dataset': 'IMDB',
        'train': train_mask.sum().item() / num_nodes,
        'val': val_mask.sum().item() / num_nodes,
        'test': test_mask.sum().item() / num_nodes
    })

    return gs, features, labels, num_classes, train_idx, val_idx, test_idx, \
            train_mask, val_mask, test_mask


def load_data(dataset, remove_self_loop=False):
    if dataset == 'ACM':
        return load_acm(remove_self_loop)
    elif dataset == 'DBLP':
        path = 'data/DBLP4057_GAT_with_idx.mat'
        return load_dblp(path, remove_self_loop)
    elif dataset == 'IMDB':
        path = 'data/imdb/'
        return load_imdb(path, remove_self_loop)
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
        model.load_state_dict(torch.load( self.filename))
