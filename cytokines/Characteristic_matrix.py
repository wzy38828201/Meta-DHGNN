import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
import scipy.sparse as sp
from Adjacency_matrix import ad

def fea():
    a = pd.read_excel(r'\all.xlsx', header=None)

    b = pd.DataFrame(pd.concat([a[0], a[1]]).unique())

    c = pd.read_excel(r'\experiments\Characteristic_matrix.xlsx')
    c0 = pd.DataFrame(c['GO'].unique())[0].tolist()

    # #Cytokines are columns and cells are rows
    # M = ad()
    # M1 = pd.DataFrame(M)
    # M1.index = b[0]
    # M1.columns = b[0]
    # ddd = pd.read_excel(r'\experiments\Cell characterization.xlsx')
    # M2 = M1[M1.index.isin(list(ddd['cell']))]
    # cyt = [i for i in b[0] if i not in list(ddd['cell'])]
    # M3 = M2[cyt]
    # M3.to_excel(r'\experiments\Cell characterization 2.xlsx')

    dd = pd.read_excel(r'\experiments\Cell characterization 2.xlsx', index_col='cell')

    c1 = pd.DataFrame(dd.columns)[0].tolist()
    c2 = pd.DataFrame((c0 + c1)).drop_duplicates() 
    # c2 = pd.concat([c0, c1], axis=0).set()
    c2.index = range(len(c2))
    # Number cytokines and cells
    dic = {}
    for i, j in zip(list(b[0]), list(b.index)):
        dic[i] = j
    # pd.DataFrame.from_dict(dic,orient='index').to_excel(r'\experiments\label_f.xlsx',header = None)

    dic1 = {}
    for i1, j1 in zip(list(c2[0]), list(c2.index)):
        dic1[i1] = j1

    d = c['labels'].str.split(',')

    # The number of the cytokine in the cytokine feature corresponds to the number of the cytokine feature
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
    
    # The English correspondence becomes the ordinal correspondence
    lis2 = []  # Number of feature
    for mm in lis1:
        if mm in dic1.keys():
            lis2.append(dic1[mm])
    f[0] = lis  # The number of cytokines
    f3 = pd.DataFrame(f[0]).drop_duplicates()
    f[1] = lis2
    f[2] = [1 for iiii in range(len(lis))]

    # Transform the characteristics of the cell into a matrix
    # dd = np.array(dd)
    dd[np.isnan(dd)] = 0
    ind = dd.index.tolist()  
    jo = [i for i in dic.keys() if i not in ind]  
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