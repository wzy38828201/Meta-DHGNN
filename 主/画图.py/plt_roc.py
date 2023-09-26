import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pandas as pd

'''画ROC曲线，计算AUC面积'''

def micro(path):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    # 计算每一类的ROC
    y_test = pd.read_excel(path + 'labels_2.xlsx', header=None)
    y_test = np.array(y_test)
    y_score = pd.read_excel(path + 'prob.xlsx')
    y_score = y_score.values
    n_classes = 3
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    return fpr,tpr,roc_auc

def macro(path):
    y_test = pd.read_excel(path + 'labels_2.xlsx', header=None)
    y_test = np.array(y_test)
    y_score = pd.read_excel(path + 'prob.xlsx')
    y_score = y_score.values
    n_classes = 3
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(3):
        print(i)
        yyy = y_test[:, i]
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    return fpr,tpr,roc_auc

def get_macro_acm_roc():
    ori_fpr, ori_tpr, ori_roc_auc = macro(r'G:\图神经网络编程\有向异构\dglhan_all_code20220914\results\原始\ACM\\')
    #att_fpr, att_tpr, att_roc_auc = macro(r'G:\图神经网络编程\有向异构\dglhan_all_code20220914\results\仅注意力\ACM\\')
    meta_fpr, meta_tpr, meta_roc_auc = macro(r'G:\图神经网络编程\有向异构\dglhan_all_code20220914\results\仅meta\ACM\\')
    adj_fpr, adj_tpr, adj_roc_auc = macro(r'G:\图神经网络编程\有向异构\dglhan_all_code20220914\results\仅adj\ACM\\')
    all_fpr, all_tpr, all_roc_auc = macro(r'G:\图神经网络编程\有向异构\dglhan_all_code20220914\results\all\ACM\\')
    print('aaaaaaaaa')
    # Plot all ROC curves
    lw=2
    plt.figure()
    plt.plot(ori_fpr["macro"], ori_tpr["macro"],
             label='dglhan (AUC = {0:0.4f})'
                   ''.format(ori_roc_auc["macro"]),
             # color='deeppink',
             linestyle='--',
             lw=lw)
    # plt.plot(att_fpr["macro"], att_tpr["macro"],
    #          label='dglhan+external-attention (AUC = {0:0.4f})'
    #                ''.format(att_roc_auc["macro"]),
    #          # color='deeppink',
    #          linestyle='--',
    #          lw=lw)
    plt.plot(adj_fpr["macro"], adj_tpr["macro"],
             label='dglhan+adj (AUC = {0:0.4f})'
                   ''.format(adj_roc_auc["macro"]),
             # color='deeppink',
             linestyle='--',
             lw=lw)
    plt.plot(meta_fpr["macro"], meta_tpr["macro"],
             label='dglhan+meta-learning (AUC = {0:0.4f})'
                   ''.format(meta_roc_auc["macro"]),
             # color='deeppink',
             linestyle='--',
             lw=lw)
    plt.plot(all_fpr["macro"], all_tpr["macro"],
             label='Meta-DHGNN (AUC = {0:0.4f})'
                   ''.format(all_roc_auc["macro"]),
             # color='deeppink',
             linestyle='--',
             lw=lw)
    # colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    # for i, color in zip(range(n_classes), colors):
    #     plt.plot(fpr[i], tpr[i], color=color, lw=lw,
    #              label='ROC curve of class {0} (area = {1:0.2f})'
    #              ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('macro-average ROC curve on the ACM dataset')
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig(r'G:\图神经网络编程\有向异构\dglhan_all_code20220914\results\ACM_macro-average_ROC.png', dpi=500)
    
def get_micro_acm_roc():
    ori_fpr,ori_tpr,ori_roc_auc = micro(r'G:\图神经网络编程\有向异构\dglhan_all_code20220914\results\原始\ACM\\')
    #att_fpr,att_tpr,att_roc_auc = micro(r'G:\图神经网络编程\有向异构\dglhan_all_code20220914\results\仅注意力\ACM\\')
    meta_fpr,meta_tpr,meta_roc_auc = micro(r'G:\图神经网络编程\有向异构\dglhan_all_code20220914\results\仅meta\ACM\\')
    adj_fpr,adj_tpr,adj_roc_auc = micro(r'G:\图神经网络编程\有向异构\dglhan_all_code20220914\results\仅adj\ACM\\')
    all_fpr,all_tpr,all_roc_auc = micro(r'G:\图神经网络编程\有向异构\dglhan_all_code20220914\results\all\ACM\\')

    # Plot all ROC curves
    lw=2
    plt.figure()
    plt.plot(ori_fpr["micro"], ori_tpr["micro"],
             label='dglhan (AUC = {0:0.2f})'
                   ''.format(ori_roc_auc["micro"]),
             # color='deeppink',
             linestyle='--',
             lw=lw)
    # plt.plot(att_fpr["micro"], att_tpr["micro"],
    #          label='dglhan+external-attention (AUC = {0:0.2f})'
    #                ''.format(att_roc_auc["micro"]),
    #          # color='deeppink',
    #          linestyle='--',
    #          lw=lw)

    plt.plot(meta_fpr["micro"], meta_tpr["micro"],
             label='dglhan+meta-learning (AUC = {0:0.2f})'
                   ''.format(meta_roc_auc["micro"]),
             # color='deeppink',
             linestyle='--',
             lw=lw)
    plt.plot(adj_fpr["micro"], adj_tpr["micro"],
             label='dglhan+adj (AUC = {0:0.2f})'
                   ''.format(adj_roc_auc["micro"]),
             # color='deeppink',
             linestyle='--',
             lw=lw)
    plt.plot(all_fpr["micro"], all_tpr["micro"],
             label='Meta-DHGNN (AUC = {0:0.2f})'
                   ''.format(all_roc_auc["micro"]),
             # color='deeppink',
             linestyle='--',
             lw=lw)

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('micro-average ROC curve on the ACM dataset')
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig(r'G:\图神经网络编程\有向异构\dglhan_all_code20220914\results\ACM_micro-average_ROC.png', dpi=500)


def get_macro_dblp_roc():
    ori_fpr, ori_tpr, ori_roc_auc = macro(r'G:\图神经网络编程\有向异构\dglhan_all_code20220914\results\原始\DBLP\\')
    #att_fpr, att_tpr, att_roc_auc = macro(r'G:\图神经网络编程\有向异构\dglhan_all_code20220914\results\仅注意力\DBLP\\')
    meta_fpr, meta_tpr, meta_roc_auc = macro(r'G:\图神经网络编程\有向异构\dglhan_all_code20220914\results\仅meta\DBLP\\')
    adj_fpr, adj_tpr, adj_roc_auc = macro(r'G:\图神经网络编程\有向异构\dglhan_all_code20220914\results\仅adj\DBLP\\')
    all_fpr, all_tpr, all_roc_auc = macro(r'G:\图神经网络编程\有向异构\dglhan_all_code20220914\results\all\DBLP\\')
    print('bbbbbbbbbb')
    # Plot all ROC curves
    lw = 2
    plt.figure()
    plt.plot(ori_fpr["macro"], ori_tpr["macro"],
             label='dglhan (AUC = {0:0.2f})'
                   ''.format(ori_roc_auc["macro"]),
             # color='deeppink',
             linestyle='--',
             lw=lw)
    # plt.plot(att_fpr["macro"], att_tpr["macro"],
    #          label='dglhan+external-attention (AUC = {0:0.2f})'
    #                ''.format(att_roc_auc["macro"]),
    #          # color='deeppink',
    #          linestyle='--',
    #          lw=lw)
    plt.plot(adj_fpr["macro"], adj_tpr["macro"],
             label='dglhan+adj (AUC = {0:0.2f})'
                   ''.format(adj_roc_auc["macro"]),
             # color='deeppink',
             linestyle='--',
             lw=lw)
    plt.plot(meta_fpr["macro"], meta_tpr["macro"],
             label='dglhan+meta-learning (AUC = {0:0.2f})'
                   ''.format(meta_roc_auc["macro"]),
             # color='deeppink',
             linestyle='--',
             lw=lw)
    plt.plot(all_fpr["macro"], all_tpr["macro"],
             label='Meta-DHGNN (AUC = {0:0.2f})'
                   ''.format(all_roc_auc["macro"]),
             # color='deeppink',
             linestyle='--',
             lw=lw)
    # colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    # for i, color in zip(range(n_classes), colors):
    #     plt.plot(fpr[i], tpr[i], color=color, lw=lw,
    #              label='ROC curve of class {0} (area = {1:0.2f})'
    #              ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('macro-average ROC curve on the DBLP dataset')
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig(r'G:\图神经网络编程\有向异构\dglhan_all_code20220914\results\DBLP_macro-average_ROC.png', dpi=500)


def get_micro_dblp_roc():
    ori_fpr, ori_tpr, ori_roc_auc = micro(r'G:\图神经网络编程\有向异构\dglhan_all_code20220914\results\原始\DBLP\\')
    #att_fpr, att_tpr, att_roc_auc = micro(r'G:\图神经网络编程\有向异构\dglhan_all_code20220914\results\仅注意力\DBLP\\')
    meta_fpr, meta_tpr, meta_roc_auc = micro(r'G:\图神经网络编程\有向异构\dglhan_all_code20220914\results\仅meta\DBLP\\')
    adj_fpr, adj_tpr, adj_roc_auc = micro(r'G:\图神经网络编程\有向异构\dglhan_all_code20220914\results\仅adj\DBLP\\')
    all_fpr, all_tpr, all_roc_auc = micro(r'G:\图神经网络编程\有向异构\dglhan_all_code20220914\results\all\DBLP\\')
    # Plot all ROC curves
    lw = 2
    plt.figure()
    plt.plot(ori_fpr["micro"], ori_tpr["micro"],
             label='dglhan (AUC = {0:0.2f})'
                   ''.format(ori_roc_auc["micro"]),
             # color='deeppink',
             linestyle='--',
             lw=lw)
    # plt.plot(att_fpr["micro"], att_tpr["micro"],
    #          label='dglhan+external-attention (AUC = {0:0.2f})'
    #                ''.format(att_roc_auc["micro"]),
    #          # color='deeppink',
    #          linestyle='--',
    #          lw=lw)

    plt.plot(meta_fpr["micro"], meta_tpr["micro"],
             label='dglhan+meta-learning (AUC = {0:0.2f})'
                   ''.format(meta_roc_auc["micro"]),
             # color='deeppink',
             linestyle='--',
             lw=lw)
    plt.plot(adj_fpr["micro"], adj_tpr["micro"],
             label='dglhan+adj (AUC = {0:0.2f})'
                   ''.format(adj_roc_auc["micro"]),
             # color='deeppink',
             linestyle='--',
             lw=lw)
    plt.plot(all_fpr["micro"], all_tpr["micro"],
             label='Meta-DHGNN (AUC = {0:0.2f})'
                   ''.format(all_roc_auc["micro"]),
             # color='deeppink',
             linestyle='--',
             lw=lw)

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('micro-average ROC curve on the DBLP dataset')
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig(r'G:\图神经网络编程\有向异构\dglhan_all_code20220914\results\DBLP_micro-average_ROC.png', dpi=500)


def get_macro_imdb_roc():
    ori_fpr, ori_tpr, ori_roc_auc = macro(r'G:\图神经网络编程\有向异构\dglhan_all_code20220914\results\原始\IMDB\\')
    #att_fpr, att_tpr, att_roc_auc = macro(r'G:\图神经网络编程\有向异构\dglhan_all_code20220914\results\仅注意力\IMDB\\')
    meta_fpr, meta_tpr, meta_roc_auc = macro(r'G:\图神经网络编程\有向异构\dglhan_all_code20220914\results\仅meta\IMDB\\')
    adj_fpr, adj_tpr, adj_roc_auc = macro(r'G:\图神经网络编程\有向异构\dglhan_all_code20220914\results\仅adj\IMDB\\')
    all_fpr, all_tpr, all_roc_auc = macro(r'G:\图神经网络编程\有向异构\dglhan_all_code20220914\results\all\IMDB\\')
    print('ccccccccccc')
    # Plot all ROC curves
    lw = 2
    plt.figure()
    plt.plot(ori_fpr["macro"], ori_tpr["macro"],
             label='dglhan (AUC = {0:0.2f})'
                   ''.format(ori_roc_auc["macro"]),
             # color='deeppink',
             linestyle='--',
             lw=lw)
    # plt.plot(att_fpr["macro"], att_tpr["macro"],
    #          label='dglhan+external-attention (AUC = {0:0.2f})'
    #                ''.format(att_roc_auc["macro"]),
    #          # color='deeppink',
    #          linestyle='--',
    #          lw=lw)
    plt.plot(adj_fpr["macro"], adj_tpr["macro"],
             label='dglhan+adj (AUC = {0:0.2f})'
                   ''.format(adj_roc_auc["macro"]),
             # color='deeppink',
             linestyle='--',
             lw=lw)
    plt.plot(meta_fpr["macro"], meta_tpr["macro"],
             label='dglhan+meta-learning (AUC = {0:0.2f})'
                   ''.format(meta_roc_auc["macro"]),
             # color='deeppink',
             linestyle='--',
             lw=lw)

    plt.plot(all_fpr["macro"], all_tpr["macro"],
             label='Meta-DHGNN (AUC = {0:0.2f})'
                   ''.format(all_roc_auc["macro"]),
             # color='deeppink',
             linestyle='--',
             lw=lw)
    # colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    # for i, color in zip(range(n_classes), colors):
    #     plt.plot(fpr[i], tpr[i], color=color, lw=lw,
    #              label='ROC curve of class {0} (area = {1:0.2f})'
    #              ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('macro-average ROC curve on the IMDB dataset')
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig(r'G:\图神经网络编程\有向异构\dglhan_all_code20220914\results\IMDB_macro-average_ROC.png', dpi=500)


def get_micro_imdb_roc():
    ori_fpr, ori_tpr, ori_roc_auc = micro(r'G:\图神经网络编程\有向异构\dglhan_all_code20220914\results\原始\IMDB\\')
    #att_fpr, att_tpr, att_roc_auc = micro(r'G:\图神经网络编程\有向异构\dglhan_all_code20220914\results\仅注意力\IMDB\\')
    meta_fpr, meta_tpr, meta_roc_auc = micro(r'G:\图神经网络编程\有向异构\dglhan_all_code20220914\results\仅meta\IMDB\\')
    adj_fpr, adj_tpr, adj_roc_auc = micro(r'G:\图神经网络编程\有向异构\dglhan_all_code20220914\results\仅adj\IMDB\\')
    all_fpr, all_tpr, all_roc_auc = micro(r'G:\图神经网络编程\有向异构\dglhan_all_code20220914\results\all\IMDB\\')
    # Plot all ROC curves
    lw = 2
    plt.figure()
    plt.plot(ori_fpr["micro"], ori_tpr["micro"],
             label='dglhan (AUC = {0:0.2f})'
                   ''.format(ori_roc_auc["micro"]),
             # color='deeppink',
             linestyle='--',
             lw=lw)
    # plt.plot(att_fpr["micro"], att_tpr["micro"],
    #          label='dglhan+external-attention (AUC = {0:0.2f})'
    #                ''.format(att_roc_auc["micro"]),
    #          # color='deeppink',
    #          linestyle='--',
    #          lw=lw)

    plt.plot(meta_fpr["micro"], meta_tpr["micro"],
             label='dglhan+meta-learning (AUC = {0:0.2f})'
                   ''.format(meta_roc_auc["micro"]),
             # color='deeppink',
             linestyle='--',
             lw=lw)
    plt.plot(adj_fpr["micro"], adj_tpr["micro"],
             label='dglhan+adj (AUC = {0:0.2f})'
                   ''.format(adj_roc_auc["micro"]),
             # color='deeppink',
             linestyle='--',
             lw=lw)
    plt.plot(all_fpr["micro"], all_tpr["micro"],
             label='dglhan+adj+meta-learning+\nexternal-attention (AUC = {0:0.2f})'
                   ''.format(all_roc_auc["micro"]),
             # color='deeppink',
             linestyle='--',
             lw=lw)

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('micro-average ROC curve on the IMDB dataset')
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig(r'G:\图神经网络编程\有向异构\dglhan_all_code20220914\results\IMDB_micro-average_ROC.png', dpi=500)

#get_macro_acm_roc()
#get_micro_acm_roc()
#get_micro_imdb_roc()
get_macro_imdb_roc()
#get_micro_dblp_roc()
#get_macro_dblp_roc()