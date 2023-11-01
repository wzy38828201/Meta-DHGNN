# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 15:29:59 2022

@author: lenovo
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve,average_precision_score

def han(score_path):
    with open(score_path, 'r') as f:
        files = f.readlines()      # 读取文件
    
    lis_all = []
    bq1 = []
    bqy1 = []
    for file in files:
        bq, bqy, s1, s2 = file.strip().split("\t")
        lis_all.append(s1)
        lis_all.append(s2)
        bq1.append(bq)
        bqy1.append(bqy)
    lis_order = sorted(set(lis_all))   # 记录所有得分情况，并去重从小到大排序，寻找各个阈值点
    
    macro_precis = []
    macro_recall = []
    
    for i in lis_order:
    
        true_p0 = 0         # 真阳
        true_n0 = 0         # 真阴
        false_p0 = 0        # 假阳
        false_n0 = 0        # 假阴
    
        true_p1 = 0
        true_n1 = 0
        false_p1 = 0
        false_n1 = 0
        
        for file in files:
            cls, pd, n0, n1= file.strip().split("\t")       # 分别计算比较各个类别的得分，分开计算，各自为二分类，
                                                                # 最后求平均，得出宏pr
    
            if float(n0) >= float(i) and cls == '0':               # 遍历所有样本，第0类为正样本，其他类为负样本，
                true_p0 = true_p0 + 1                              # 大于等于阈值，并且真实为正样本，即为真阳，
            elif float(n0) >= float(i) and cls != '0':             # 大于等于阈值，真实为负样本，即为假阳；
                false_p0 = false_p0 + 1                            # 小于阈值，真实为正样本，即为假阴
            elif float(n0) < float(i) and cls == '0':
                false_n0 = false_n0 + 1
    
            if float(n1) >= float(i) and cls == '1':                # 遍历所有样本，第1类为正样本，其他类为负样本
                true_p1 = true_p1 + 1
            elif float(n1) >= float(i) and cls != '1':
                false_p1 = false_p1 + 1
            elif float(n1) < float(i) and cls == '1':
                false_n1 = false_n1 + 1

    
        prec0 = (true_p0+0.00000000001) / (true_p0 + false_p0 + 0.00000000001)      # 计算各类别的精确率，小数防止分母为0
        prec1 = (true_p1+0.00000000001) / (true_p1 + false_p1 + 0.00000000001)
        
        recall0 = (true_p0+0.00000000001)/(true_p0+false_n0 + 0.00000000001)        # 计算各类别的召回率，小数防止分母为0
        recall1 = (true_p1+0.00000000001) / (true_p1 + false_n1+0.00000000001)
    
        precision = (prec0 + prec1)/2
        recall = (recall0 + recall1)/2             # 多分类求得平均精确度和平均召回率，即宏macro_pr
        macro_precis.append(precision)
        macro_recall.append(recall)
    
    bq10 = []
    bq11 = []
    for ui0 in bq1:
        if str(ui0) == '0':
            bq10.append(int(ui0))
        else:
            bq10.append(1)
    for ui1 in bq1:
        if str(ui1) == '1':
            bq11.append(int(ui1))
        else:
            bq11.append(0)
    
    bqy10 = []
    bqy11 = []
    for ui00 in bqy1:
        if str(ui00) == '0':
            bqy10.append(int(ui00))
        else:
            bqy10.append(1)
    for ui11 in bqy1:
        if str(ui11) == '1':
            bqy11.append(int(ui11))
        else:
            bqy11.append(0)
    
    PRC0 = average_precision_score(bq10,bqy10)
    PRC1 = average_precision_score(bq11,bqy11)
    
    #    print(PRC0)
    #    print(PRC1)
    #    print(PRC2)
    print((PRC0+PRC1)/2)
    PRC = (PRC0+PRC1)/2
    
    macro_precis.append(1)
    macro_recall.append(0)
    #print(macro_precis)
    #print(macro_recall)
    return PRC,macro_recall,macro_precis

# score_path1 = r"G:\图神经网络编程\有向异构\dglhan_all_code20220914\results\原始\ACM\pr曲线.txt"  # 文件路径
# score_path2 = r"G:\图神经网络编程\有向异构\dglhan_all_code20220914\results\仅adj\ACM\pr曲线.txt"  # 文件路径
# score_path3 = r"G:\图神经网络编程\有向异构\dglhan_all_code20220914\results\仅meta\ACM\pr曲线.txt"  # 文件路径
# score_path5 = r"G:\图神经网络编程\有向异构\dglhan_all_code20220914\results\all\ACM\pr曲线.txt"  # 文件路径

# score_path1 = r"G:\图神经网络编程\有向异构\dglhan_all_code20220914\results\原始\DBLP\pr曲线.txt"  # 文件路径
# score_path2 = r"G:\图神经网络编程\有向异构\dglhan_all_code20220914\results\仅adj\DBLP\pr曲线.txt"  # 文件路径
# score_path3 = r"G:\图神经网络编程\有向异构\dglhan_all_code20220914\results\仅meta\DBLP\pr曲线.txt"  # 文件路径
# score_path5 = r"G:\图神经网络编程\有向异构\dglhan_all_code20220914\results\all\DBLP\pr曲线.txt"  # 文件路径

score_path1 = "cytokines/Results/original/pr.txt"  # 文件路径，先生成prob文件
score_path2 = "cytokines/Results/adj/pr.txt"  # 文件路径，先生成prob文件
score_path3 = "cytokines/Results/meta/pr.txt"  # 文件路径，先生成prob文件
score_path4 = "cytokines/Results/Meta-GNN\pr.txt"  # 文件路径，先生成prob文件
score_path5 = "cytokines/Results/all/pr.txt"  # 文件路径，先生成prob文件

PRC1,macro_recall1,macro_precis1 = han(score_path1)
PRC2,macro_recall2,macro_precis2 = han(score_path2)
PRC3,macro_recall3,macro_precis3 = han(score_path3)
PRC4,macro_recall4,macro_precis4 = han(score_path4)
PRC5,macro_recall5,macro_precis5 = han(score_path5)

plt.figure(figsize=(10,6))
plt.step(macro_recall4, macro_precis4, color='orange', label=' Meta-GNN (PRC={:.4f})'.format(PRC4))
plt.step(macro_recall1, macro_precis1, color='red', label=' dglhan (PRC={:.4f})'.format(PRC1))
plt.step(macro_recall2, macro_precis2, color='black', label=' dglhan+adj (PRC={:.4f})'.format(PRC2))
plt.step(macro_recall3, macro_precis3, color='green', label=' dglhan+meta-learning (PRC={:.4f})'.format(PRC3))
plt.step(macro_recall5, macro_precis5, color='purple', label=' Meta-DHGNN (PRC={:.4f})'.format(PRC5))


#x1 = np.array(macro_recall1)
#y1 = np.array(macro_precis1)
#x2 = np.array(macro_recall2)
#y2 = np.array(macro_precis2)
#x3 = np.array(macro_recall3)
#y3 = np.array(macro_precis3)
#x4 = np.array(macro_recall4)
#y4 = np.array(macro_precis4)
#x5 = np.array(macro_recall5)
#y5 = np.array(macro_precis5)

plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('recall')
plt.ylabel('precision')
plt.title('PR curve')
#plt.plot([0, 1], [1, 0], color='m', linestyle='--')

plt.legend(title='')
#plt.plot(x1, y1,color='grey', alpha=0.3)
#plt.plot(x2, y2,color='grey', alpha=0.3)
#plt.plot(x3, y3,color='grey', alpha=0.3)
#plt.plot(x4, y4,color='grey', alpha=0.3)
#plt.plot(x5, y5,color='grey', alpha=0.3)
#plt.legend(title='')

plt.savefig('cytokines-PRC.png',dpi=300)
plt.show()
#plt.savefig('ROC curves.pdf',dpi=300) 
#plt.savefig('ROC curves.png',dpi=300)





