import torch
import argparse

def get_citation_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')#16
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')#0.5
    parser.add_argument('--dataset', type=str, default='PPI', help='Dataset to use.')#cora
    parser.add_argument('--model', type=str, default='GCN', help='Model to use.')#GCN
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disable CUDA training.')#False
    parser.add_argument('--n_way', type=int, default=2, help='Classes want to be classify')  # 2
    parser.add_argument('--normalization', type=str, default='FirstOrderGCN',
                        help='Normalization method for the adjacency matrix.')  #
    args, _ = parser.parse_known_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args
