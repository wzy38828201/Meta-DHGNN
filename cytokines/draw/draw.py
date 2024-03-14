import matplotlib.pyplot as plt
import numpy as np
'''画loss曲线图'''
def acm():
    train_losses_ori = []
    #train_losses_adj_meta = []
    train_losses_att = []
    train_losses_all = []

    # eval_losses = []
    with open(r'G:\图神经网络编程\有向异构\dglhan改实验\结果\原始\细胞因子loss.txt', 'r', encoding='utf-8') as acmfile:
        acmlines = acmfile.readlines()
        for acml in acmlines:
            train_losses_ori.append(float(acml.split(' ')[5]))
            # eval_losses.append(float(acml.split(' ')[19]))

    with open(r'G:\图神经网络编程\有向异构\dglhan改实验\结果\仅注意力\细胞因子loss.txt', 'r', encoding='utf-8') as acmfile:
        acmlines = acmfile.readlines()
        for acml in acmlines:
            train_losses_att.append(float(acml.split(' ')[5]))

    with open(r'G:\图神经网络编程\有向异构\dglhan改实验\结果\all\细胞因子loss.txt', 'r', encoding='utf-8') as acmfile:
        acmlines = acmfile.readlines()
        for acml in acmlines:
            train_losses_all.append(float(acml.split(' ')[5]))
    #
    train_losses_adj = []
    train_losses_meta = []
    with open(r'G:\图神经网络编程\有向异构\dglhan改实验\结果\仅adj\细胞因子loss.txt', 'r', encoding='utf-8') as acmfile:
        acmlines = acmfile.readlines()
        for acml in acmlines:
            train_losses_adj.append(float(acml.split(' ')[5]))

    with open(r'G:\图神经网络编程\有向异构\dglhan改实验\结果\仅meta\细胞因子loss.txt', 'r', encoding='utf-8') as acmfile:
        acmlines = acmfile.readlines()
        for acml in acmlines:
            train_losses_meta.append(float(acml.split(' ')[5]))

    # print(len(train_losses_ori))
    # print(len(train_losses_adj_meta))
    # print(len(train_losses_att))
    # print(len(train_losses_all))
    # print(eval_losses)

    plt.plot(np.arange(len(train_losses_ori)), train_losses_ori, label="dglhan")
    plt.plot(np.arange(len(train_losses_att)), train_losses_att, label="dglhan+external-attention")
    plt.plot(np.arange(len(train_losses_adj)), train_losses_adj, label="dglhan+adj")
    plt.plot(np.arange(len(train_losses_meta)), train_losses_meta, label="dglhan+meta-learning")
    # plt.plot(np.arange(len(train_losses_adj_meta)), train_losses_adj_meta, label="dglhan+adj+meta-learning")
    plt.plot(np.arange(len(train_losses_all)), train_losses_all, label="Meta-DHGNN")
    # plt.plot(np.arange(len(eval_losses)), eval_losses, label="valid loss")
    plt.legend()  # 显示图例
    plt.xlabel('Epoches')
    # plt.ylabel("epoch")
    plt.title('train loss on the Cytokines dataset ')
    plt.savefig(r'G:\图神经网络编程\有向异构\dglhan改实验\结果\细胞因子.png', dpi=500)
    plt.close()

acm()