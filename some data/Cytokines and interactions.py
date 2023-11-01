# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 13:26:32 2022

@author: lenovo
"""

import pandas as pd
from scipy.sparse import coo_matrix

a = pd.read_excel(r'G:\图神经网络编程\有向异构\dglhan改\实验\string_interactions.xlsx',header = None)
c = pd.read_excel(r'G:\图神经网络编程\有向异构\dglhan改\实验\细胞因子名称.xlsx',header = None)
a1 = a[0].tolist()
a2 = a[1].tolist()
a3 = a1+a2
a4 = a[2].tolist()
b = pd.Series(a3).unique().tolist()
#d = c[3].tolist()
#e = [i for i in d if i not in b]
#for i,j in enumerate():
d = dict(zip(c[1],c[0]))
f = []
f1 = []
for i in a1:
    if i in d.keys():
        f.append(d[i])
for ii in a2:
    if ii in d.keys():
        f1.append(d[ii])
a_ = pd.DataFrame()
a_[0] = f
a_[1] = f1
a_[2] = a4
#构造邻接矩阵
M = coo_matrix((a_.iloc[:,2], (a_.iloc[:,0],a_.iloc[:,1])), shape=(len(c), len(c))).toarray()