# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 20:28:00 2020

@author: lenovo
"""
import pandas as pd

a = pd.read_excel(r'C:\Users\lenovo\Desktop\随机取样.xlsx')
b = pd.read_excel(r'C:\Users\lenovo\Desktop\原来.xlsx')
s = a['细胞因子'].tolist()
d = b['细胞因子'].tolist()
ret1 = list(set(s).union(set(d)))#并集
ret2 = list(set(s).intersection(set(d)))#交集
list1 = []#983存在，1765里不存在的
list2 = []#1765存在，983里不存在的
for i in s:
    if i not in d:
        list1.append(i)
    else:
        continue
for m in d:
    if m not in s:
        list2.append(m)
    else:
        continue