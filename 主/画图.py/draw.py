import matplotlib.pyplot as plt
import numpy as np
'''画loss曲线图'''
def acm():
    train_losses_ori = []
    train_losses_adj_meta = []
    train_losses_att = []
    train_losses_all = []

    # eval_losses = []
    with open('00origin-bestresults/acmloss.txt', 'r', encoding='utf-8') as acmfile:
        acmlines = acmfile.readlines()
        for acml in acmlines:
            train_losses_ori.append(float(acml.split(' ')[5]))
            # eval_losses.append(float(acml.split(' ')[19]))

    with open('01adj-meta-bestresults/acmloss.txt', 'r', encoding='utf-8') as acmfile:
        acmlines = acmfile.readlines()
        for acml in acmlines:
            train_losses_adj_meta.append(float(acml.split(' ')[5]))

    with open('02onlyatt-bestresults/acmloss.txt', 'r', encoding='utf-8') as acmfile:
        acmlines = acmfile.readlines()
        for acml in acmlines:
            train_losses_att.append(float(acml.split(' ')[5]))

    with open('03all-bestresults/acmloss.txt', 'r', encoding='utf-8') as acmfile:
        acmlines = acmfile.readlines()
        for acml in acmlines:
            train_losses_all.append(float(acml.split(' ')[5]))

    train_losses_adj = []
    train_losses_meta = []
    with open('04onlyadj/acmloss.txt', 'r', encoding='utf-8') as acmfile:
        acmlines = acmfile.readlines()
        for acml in acmlines:
            train_losses_adj.append(float(acml.split(' ')[5]))

    with open('05onlymeta/acmloss.txt', 'r', encoding='utf-8') as acmfile:
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
    plt.plot(np.arange(len(train_losses_all)), train_losses_all, label="dglhan+adj+meta-learning+external-attention")
    # plt.plot(np.arange(len(eval_losses)), eval_losses, label="valid loss")
    plt.legend()  # 显示图例
    plt.xlabel('Epoches')
    # plt.ylabel("epoch")
    plt.title('train loss on the ACM dataset ')
    plt.savefig('acm.png', dpi=500)
    plt.close()


def dblp():
    train_losses_ori = []
    train_losses_adj_meta = []
    train_losses_att = []
    train_losses_all = []
    # eval_losses = []
    with open('00origin-bestresults/dblploss.txt', 'r', encoding='utf-8') as acmfile:
        acmlines = acmfile.readlines()
        for acml in acmlines:
            train_losses_ori.append(float(acml.split(' ')[5]))
            # eval_losses.append(float(acml.split(' ')[19]))

    with open('01adj-meta-bestresults/dblploss.txt', 'r', encoding='utf-8') as acmfile:
        acmlines = acmfile.readlines()
        for acml in acmlines:
            train_losses_adj_meta.append(float(acml.split(' ')[5]))

    with open('02onlyatt-bestresults/dblploss.txt', 'r', encoding='utf-8') as acmfile:
        acmlines = acmfile.readlines()
        for acml in acmlines:
            train_losses_att.append(float(acml.split(' ')[5]))

    with open('03all-bestresults/dblploss.txt', 'r', encoding='utf-8') as acmfile:
        acmlines = acmfile.readlines()
        for acml in acmlines:
            train_losses_all.append(float(acml.split(' ')[5]))

    train_losses_adj = []
    train_losses_meta = []
    with open('04onlyadj/dblploss.txt', 'r', encoding='utf-8') as acmfile:
        acmlines = acmfile.readlines()
        for acml in acmlines:
            train_losses_adj.append(float(acml.split(' ')[5]))

    with open('05onlymeta/dblploss.txt', 'r', encoding='utf-8') as acmfile:
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

    plt.plot(np.arange(len(train_losses_all)), train_losses_all, label="dglhan+adj+meta-learning+external-attention")
    # plt.plot(np.arange(len(eval_losses)), eval_losses, label="valid loss")
    plt.ylim(0, 2)
    plt.legend()  # 显示图例
    plt.xlabel('Epoches')
    # plt.ylabel("epoch")
    plt.title('train loss on the DBLP dataset')
    plt.savefig('dblp.png', dpi=500)
    plt.close()

def imdb():
    train_losses_ori = []
    train_losses_adj_meta = []
    train_losses_att = []
    train_losses_all = []
    # eval_losses = []
    with open('00origin-bestresults/imdbloss.txt', 'r', encoding='utf-8') as acmfile:
        acmlines = acmfile.readlines()
        for acml in acmlines:
            train_losses_ori.append(float(acml.split(' ')[5]))
            # eval_losses.append(float(acml.split(' ')[19]))

    with open('01adj-meta-bestresults/imdbloss.txt', 'r', encoding='utf-8') as acmfile:
        acmlines = acmfile.readlines()
        for acml in acmlines:
            train_losses_adj_meta.append(float(acml.split(' ')[5]))

    with open('02onlyatt-bestresults/imdbloss.txt', 'r', encoding='utf-8') as acmfile:
        acmlines = acmfile.readlines()
        for acml in acmlines:
            train_losses_att.append(float(acml.split(' ')[5]))

    with open('03all-bestresults/imdbloss.txt', 'r', encoding='utf-8') as acmfile:
        acmlines = acmfile.readlines()
        for acml in acmlines:
            train_losses_all.append(float(acml.split(' ')[5]))

    train_losses_adj = []
    train_losses_meta = []
    with open('04onlyadj/imdbloss.txt', 'r', encoding='utf-8') as acmfile:
        acmlines = acmfile.readlines()
        for acml in acmlines:
            train_losses_adj.append(float(acml.split(' ')[5]))

    with open('05onlymeta/imdbloss.txt', 'r', encoding='utf-8') as acmfile:
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

    plt.plot(np.arange(len(train_losses_all)), train_losses_all, label="dglhan+adj+meta-learning+external-attention")
    # plt.plot(np.arange(len(eval_losses)), eval_losses, label="valid loss")
    plt.ylim(0, 3)
    plt.legend()  # 显示图例
    plt.xlabel('Epoches')
    # plt.ylabel("epoch")
    plt.title('train loss on the IMDB dataset')
    plt.savefig('imdb.png', dpi=500)
acm()
dblp()
imdb()