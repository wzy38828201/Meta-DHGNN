import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import GATConv

class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)                    # (M, 1)
        beta = torch.softmax(w, dim=0)                 # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape) # (N, M, 1)

        return (beta * z).sum(1)                       # (N, D * K)

class HANLayer(nn.Module):
    """
    HAN layer.

    Arguments
    ---------
    num_meta_paths : number of homogeneous graphs generated from the metapaths.
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability

    Inputs
    ------
    g : list[DGLGraph]
        List of graphs
    h : tensor
        Input features

    Outputs
    -------
    tensor
        The output feature
    """
    def __init__(self, num_meta_paths, in_size, out_size, layer_num_heads, dropout):
        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(num_meta_paths):
            self.gat_layers.append(GATConv(in_size, out_size, layer_num_heads,
                                           dropout, dropout, activation=F.elu))
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        self.num_meta_paths = num_meta_paths

    def forward(self, gs, h):
        semantic_embeddings = []

        for i, g in enumerate(gs):
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            semantic_embeddings.append(self.gat_layers[i](g, h).flatten(1))
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)                  # (N, M, D * K)

        return self.semantic_attention(semantic_embeddings)                            # (N, D * K)

class HAN(nn.Module):
    def __init__(self, num_meta_paths, in_size, hidden_size, out_size, num_heads, dropout):
        super(HAN, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(num_meta_paths, in_size, hidden_size, num_heads[0], dropout))
        for l in range(1, len(num_heads)):
            self.layers.append(HANLayer(num_meta_paths, hidden_size * num_heads[l-1],
                                        hidden_size, num_heads[l], dropout))
        self.predict = nn.Linear(hidden_size * num_heads[-1], out_size)

    def forward(self, g, h):
        for gnn in self.layers:
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            h = gnn(g, h)
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
        return self.predict(h)



class Res_HAN(nn.Module):
    def __init__(self, num_meta_paths, in_size, hidden_size, out_size, num_heads, dropout):
        super(Res_HAN, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(num_meta_paths, hidden_size * num_heads[-1], hidden_size, num_heads[0], dropout))
        for l in range(1, len(num_heads)):
            self.layers.append(HANLayer(num_meta_paths, hidden_size * num_heads[l-1],
                                        hidden_size, num_heads[l], dropout))
        self.input_fc = nn.Linear(in_size, hidden_size * num_heads[-1])
        self.predict = nn.Linear(hidden_size * num_heads[-1], out_size)
        self.res_coefficient = 0.2
        self.last_layer_coefficient = 0.5
        self.res_weights = torch.nn.Parameter(torch.randn((len(num_heads))))

    def forward(self, g, h):
        h = self.input_fc(h)
        h_input = h
        layer_h_out = []        # Save h for each layer
        layer_i = 0
        for gnn in self.layers:
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            h = gnn(g, h)
            if layer_i == 0:
                h = h + self.res_coefficient * h_input
            else:
                h = h + self.res_coefficient * h_input + self.last_layer_coefficient * layer_h_out[layer_i-1]

            layer_h_out.append(h)

            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()

        weight = F.softmax(self.res_weights, dim=0)
        for i in range(len(layer_h_out)):
            layer_h_out[i] = layer_h_out[i] * weight[i]

        h = sum(layer_h_out)

        return self.predict(h)
