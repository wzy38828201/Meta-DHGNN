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
    n_classes = 2
    for i in range(2):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Compute micro-average ROC curve and ROC area
    ooo = y_test.ravel()
    ooo1 = y_score.ravel()
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    return fpr,tpr,roc_auc

def macro(path):
    y_test = pd.read_excel(path + 'labels_2.xlsx', header=None)
    y_test = np.array(y_test)
    y_score = pd.read_excel(path + 'prob.xlsx')
    y_score = y_score.values
    n_classes = 2
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(2):
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
    att_fpr, att_tpr, att_roc_auc = macro(r'G:\图神经网络编程\有向异构\dglhan改实验\结果\Meta-GNN\\')
    ori_fpr, ori_tpr, ori_roc_auc = macro(r'G:\图神经网络编程\有向异构\dglhan改实验\结果\原始\\')
    meta_fpr, meta_tpr, meta_roc_auc = macro(r'G:\图神经网络编程\有向异构\dglhan改实验\结果\仅meta\\')
    adj_fpr, adj_tpr, adj_roc_auc = macro(r'G:\图神经网络编程\有向异构\dglhan改实验\结果\仅adj\\')
    all_fpr, all_tpr, all_roc_auc = macro(r'G:\图神经网络编程\有向异构\dglhan改实验\结果\all\\')
    # Plot all ROC curves
    lw=2
    plt.figure()
    plt.plot(att_fpr["macro"], att_tpr["macro"],
             label='Meta-GNN (AUC = {0:0.2f})'
                   ''.format(att_roc_auc["macro"]),
             color='orange',
             linestyle='-',
             lw=lw)

    plt.plot(ori_fpr["macro"], ori_tpr["macro"],
             label='dglhan (AUC = {0:0.2f})'
                   ''.format(ori_roc_auc["macro"]),
             color='blue',
             linestyle='-',
             lw=lw)

    plt.plot(meta_fpr["macro"], meta_tpr["macro"],
             label='dglhan+meta-learning (AUC = {0:0.2f})'
                   ''.format(meta_roc_auc["macro"]),
             color='green',
             linestyle='-',
             lw=lw)
    plt.plot(adj_fpr["macro"], adj_tpr["macro"],
             label='dglhan+adj (AUC = {0:0.2f})'
                   ''.format(adj_roc_auc["macro"]),
             color='red',
             linestyle='-',
             lw=lw)
    plt.plot(all_fpr["macro"], all_tpr["macro"],
             label='Meta-DHGNN (AUC = {0:0.2f})'
                   ''.format(all_roc_auc["macro"]),
             color='deeppink',
             linestyle='-',
             lw=lw)

    # colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    # for i, color in zip(range(n_classes), colors):
    #     plt.plot(fpr[i], tpr[i], color=color, lw=lw,
    #              label='ROC curve of class {0} (area = {1:0.2f})'
    #              ''.format(i, roc_auc[i]))

    font = {'family': 'Times New Roman', 'weight': 'bold', 'size': 15}
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate',fontdict=font)
    plt.ylabel('True Positive Rate',fontdict=font)
    plt.xticks(fontproperties = 'Times New Roman', size = 20)
    plt.yticks(fontproperties = 'Times New Roman', size = 20)
    
    plt.title('Macro-average ROC curve on the cytokines',fontdict=font)
    plt.legend(loc="lower left",prop=font)
    # plt.show()
    plt.savefig(r'G:\图神经网络编程\有向异构\dglhan改实验\结果\细胞因子_macro-average_ROC.png', dpi=600,bbox_inches='tight')
    plt.savefig(r'G:\图神经网络编程\有向异构\dglhan改实验\结果\细胞因子_macro-average_ROC.pdf', dpi=600,bbox_inches='tight', format="pdf")

get_macro_acm_roc()