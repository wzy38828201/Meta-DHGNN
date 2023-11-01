import pandas as pd
#import numpy as np
from scipy.sparse import coo_matrix
def ad():
    a = pd.read_excel('Total (without cytokines).xlsx',header= None)

    b = pd.DataFrame(pd.concat([a[0],a[1]]).unique())

    ##邻接矩阵
    #将细胞因子和细胞编上序号
    dic = {}
    for i,j in zip(list(b[0]),list(b.index)):
        dic[i] = j

    #细胞因子和细胞中文对应关系变成序号对应关系
    da = pd.DataFrame()
    li = []
    for m in a[0]:
        if m in dic.keys():
            li.append(dic[m])
    li1 = []
    for m in a[1]:
        if m in dic.keys():
            li1.append(dic[m])
    da[0] = li
    da[1] = li1

    #构造邻接矩阵
    M = coo_matrix((a.iloc[:,2], (da.iloc[:,0],da.iloc[:,1])), shape=(len(b), len(b))).toarray()

    return M




