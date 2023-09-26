import torch
import argparse

def get_citation_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_shot_0', type=int, default=5, help='How many shot during meta-train')#20
    parser.add_argument('--train_shot_1', type=int, default=5, help='How many shot during meta-train')#20
    parser.add_argument('--test_shot_0', type=int, default=51, help='How many shot during meta-test')#1
    parser.add_argument('--test_shot_1', type=int, default=20, help='How many shot during meta-test')#1
    parser.add_argument('--n_way', type=int, default=2, help='Classes want to be classify')#2
    parser.add_argument('--step', type=int, default=20, help='How many times to random select node to test')#50
    parser.add_argument('--step1', type=int, default=1, help='How many times to random select node to test')#100
    parser.add_argument('--node_num', type=int, default=2708, help='Node number (dataset)')#2708
    parser.add_argument('--iteration', type=int, default=5, help='Iteration each cross_validation')#50,现在取10
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disable CUDA training.')#False
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')#42
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train.')#200要训练的epoch的数目，现在取20
    parser.add_argument('--epochs1', type=int, default=20, help='Number of epochs to train.')  # 200要训练的epoch的数目，现在取20
    parser.add_argument('--lr', type=float, default=0.0005, help='Initial learning rate.')#0.0001
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')#5e-4
    parser.add_argument('--hidden', type=int, default=10, help='Number of hidden units.')#16
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')#0.5
    parser.add_argument('--dataset', type=str, default='PPI', help='Dataset to use.')#cora
    parser.add_argument('--model', type=str, default='GCN', help='Model to use.')#GCN
    parser.add_argument('--normalization', type=str, default='FirstOrderGCN', help='Normalization method for the adjacency matrix.')#FirstOrderGCN
    parser.add_argument('--degree', type=int, default=2, help='degree of the approximation.')#2

    args, _ = parser.parse_known_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args
