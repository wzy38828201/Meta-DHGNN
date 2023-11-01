import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
import scipy.sparse as sp
from 邻接矩阵 import ad

def fea():
    a = pd.read_excel('cytokines/code/Total (without cytokines).xlsx', header=None)

    b = pd.DataFrame(pd.concat([a[0], a[1]]).unique())

    c = pd.read_excel('cytokines/code/feature matrix.xlsx')
    c0 = pd.DataFrame(c['GO'].unique())[0].tolist()

    # #细胞因子为列，细胞为行的特征矩阵
    # M = ad()
    # M1 = pd.DataFrame(M)
    # M1.index = b[0]
    # M1.columns = b[0]
    # ddd = pd.read_excel(r'C:\Users\lenovo\Desktop\SCI\第三篇SCI\自己整理的\细胞特征.xlsx')
    # M2 = M1[M1.index.isin(list(ddd['细胞']))]
    # cyt = [i for i in b[0] if i not in list(ddd['细胞'])]
    # M3 = M2[cyt]
    # M3.to_excel(r'C:\Users\lenovo\Desktop\SCI\第三篇SCI\自己整理的\细胞特征新.xlsx')

    dd = pd.read_excel('cytokines/cell features new.xlsx', index_col='cell')

    c1 = pd.DataFrame(dd.columns)[0].tolist()
    c2 = pd.DataFrame((c0 + c1)).drop_duplicates()  # 细胞特征去重
    # c2 = pd.concat([c0, c1], axis=0).set()
    c2.index = range(len(c2))
    # 将细胞因子和细胞编上序号
    dic = {}
    for i, j in zip(list(b[0]), list(b.index)):
        dic[i] = j
    # pd.DataFrame.from_dict(dic,orient='index').to_excel(r'C:\Users\lenovo\Desktop\SCI\第三篇SCI\自己整理的\标签.xlsx',header = None)

    dic1 = {}
    for i1, j1 in zip(list(c2[0]), list(c2.index)):
        dic1[i1] = j1

    d = c['labels'].str.split(',')

    # 将细胞因子特征中的细胞因子序号和细胞因子特征的序号对应起来
    f = pd.DataFrame()
    lis = []
    lis1 = []
    lis3 = []
    for k, o in enumerate(c2[0]):
        for ii, jj in enumerate(d):
            if k == ii:
                for mm in jj:
                    if mm in dic.keys():
                        lis.append(dic[mm])
                        lis1.append(o)
                        lis3.append(mm)

    lis3 = pd.DataFrame(lis3).drop_duplicates()
    # 英文对应关系变成序号对应关系
    lis2 = []  # 特征的编号
    for mm in lis1:
        if mm in dic1.keys():
            lis2.append(dic1[mm])
    f[0] = lis  # 细胞因子的编号
    f3 = pd.DataFrame(f[0]).drop_duplicates()
    f[1] = lis2
    f[2] = [1 for iiii in range(len(lis))]

    # 将细胞的特征给转化为矩阵
    # dd = np.array(dd)
    dd[np.isnan(dd)] = 0
    ind = dd.index.tolist()  # 细胞有哪些
    jo = [i for i in dic.keys() if i not in ind]  # 细胞因子有哪些
    list1 = []
    list2 = []
    list3 = []
    for iii in dd:
        for jjj, ooo in enumerate(dd[iii]):
            if ooo == 1:
                list1.append(ind[jjj])
                list2.append(iii)
                list3.append(1)
            else:
                list1.append(ind[jjj])
                list2.append(iii)
                list3.append(0)
    f1 = pd.DataFrame()
    list1_ = []
    list2_ = []
    for fi in list1:
        if fi in dic.keys():
            list1_.append(dic[fi])
    for fj in list2:
        if fj in dic1.keys():
            list2_.append(dic1[fj])
    f1[0] = list1_
    f4 = pd.DataFrame(f1[0]).drop_duplicates()
    f1[1] = list2_
    f1[2] = list3

    fz = pd.concat([f, f1], axis=0)
    f5 = pd.DataFrame(fz[0]).drop_duplicates()
    M11 = coo_matrix((fz.iloc[:, 2], (fz.iloc[:, 0], fz.iloc[:, 1]))).toarray()

    return M11