# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 17:11:43 2020

@author: lenovo
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 20:38:53 2020

@author: lenovo
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 17:14:27 2020

@author: Wei Zhenyu
"""

import pandas as pd
import numpy as np

def biaoqianfenlei():
    #把标签分类好
    #把正负标签的类别加载到总标签里面去
    protein_map= pd.read_csv(open(r'G:\元学习\Meta-GNN-master\gcn_fewshot\1769\全部的细胞因子节点映射.csv'),index_col=0)
    direct = r'G:\元学习\Meta-GNN-master\gcn_fewshot\1769'
    #正标签
    def label_num_1(len_1):
        a = []
        for i in range(len_1):
            if i == 1:
                a.append(i)
            elif i!=1:
                i = 1
                a.append(i)
        return a
    #print(a)

    label_1 = pd.read_csv(open(direct+r'\label正.txt'),header=None,sep='\t',usecols=[0])
    len_1 = len(label_1)
    #print(len_1)
    label_1_1 = protein_map.loc[label_1[0]]
    label_1_v = label_1_1['nodes'].values
    label_num= pd.read_csv(open(r'G:\元学习\Meta-GNN-master\gcn_fewshot\1769\标签分类\标签的类别.csv'),header=None,index_col=False)
    label_num1 = np.c_[label_num.loc[label_1_v],label_num_1(len_1)]
    s = pd.DataFrame(label_num1)
    #print(s)
    s.to_csv(r'G:\元学习\Meta-GNN-master\gcn_fewshot\1769\标签分类\正标签分类完成.csv')

    #负标签
    def label_num_0(len_0):
        b = []
        for i in range(len_0):
            if i == 0:
                b.append(i)
            elif i!=0:
                i = 0
                b.append(i)
        return b

    label_0 = pd.read_csv(open(direct+r'\label负.txt'),header=None,sep='\t',usecols=[0])
    len_0 = len(label_0)
    #print(label_0[0].values)
    label_0_1 = protein_map.loc[label_0[0]]
    label_0_v = label_0_1['nodes'].values
    label_num0 = np.c_[label_num.loc[label_0_v],label_num_0(len_0)]
    d = pd.DataFrame(label_num0)
    d.to_csv(r'G:\元学习\Meta-GNN-master\gcn_fewshot\1769\标签分类\负标签分类完成.csv')


    dictionary = {}
    label_len =len(pd.read_csv(r'G:\元学习\Meta-GNN-master\gcn_fewshot\1769\标签分类\标签的类别.csv',header=None))
    #print(label_len)
    for i in range(label_len):
        dictionary[i]=2
    #print(dictionary.keys())
    label_z = pd.read_csv(r'G:\元学习\Meta-GNN-master\gcn_fewshot\1769\标签分类\正标签分类完成.csv',index_col=0)
    label_f = pd.read_csv(r'G:\元学习\Meta-GNN-master\gcn_fewshot\1769\标签分类\负标签分类完成.csv',index_col=0)
    z_l = label_z['0'].values
    f_l = label_f['0'].values
    #print(z_l)
    for z in z_l:
        dictionary[z]=1
    for f in f_l:
        dictionary[f]=0
    #print(dictionary)
    dictionary_1 = pd.DataFrame.from_dict(dictionary,orient ='index')
    #print(dictionary_1)
    dictionary_1.to_csv(r'G:\元学习\Meta-GNN-master\gcn_fewshot\1769\标签分类\全部的因子分类.csv')