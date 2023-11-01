import torch
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_geometric.MessagePassing import MessagePassing
from torch_geometric.utils import add_remaining_self_loops, add_self_loops
from torch_geometric.utils import to_undirected, is_undirected
from torch_geometric.inits import glorot, zeros

class DIGCNConv(MessagePassing):#有向的变化
    r"""The graph convolutional operator takes from Pytorch Geometric.图卷积算子取自Pytorch geometry。
    The spectral operation is the same with Kipf's GCN.谱运算与Kipf的GCN相同
    DiGCN preprocesses the adjacency matrix and does not require a norm operation during the convolution operation.
    DiGCN对邻接矩阵进行预处理，在卷积操作期间不需要范数操作。
    Args:
        in_channels (int): Size of each input sample.in_channels (int):每个输入样例的大小。
        out_channels (int): Size of each output sample.out_channels (int):每个输出示例的大小。
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the adj matrix on first execution, and will use the
            cached version for further executions.如果设置为:obj: ' True '，该层将在第一次执行时缓存adj矩阵，并在后续执行时使用缓存的版本。
            Please note that, all the normalized adj matrices (including undirected)
            are calculated in the dataset preprocessing to reduce time comsume.请注意，所有归一化的adj矩阵(包括无向的)均计算在数据集预处理，以减少时间消耗。
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)在转换中，该参数只能设置为:obj: ' True '学习的场景。(默认::obj:“假”)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)bias (bool，可选):如果设置为:obj: ' False '，该层将不会学习一种添加剂的偏见。(默认::obj:“真正的”)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels, out_channels, improved=False, cached=True,
                 bias=True, **kwargs):
        super(DIGCNConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None
        self.cached_num_edges = None
    
    def forward(self, x, edge_index, edge_weight=None):
        """"""
        x = torch.matmul(x, self.weight)

        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            if edge_weight is None:
                raise RuntimeError(
                    'Normalized adj matrix cannot be None. Please '
                    'obtain the adj matrix in preprocessing.')
            else:
                norm = edge_weight
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j if norm is not None else x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
