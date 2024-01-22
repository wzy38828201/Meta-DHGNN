import pandas as pd
import numpy as np

a = pd.read_excel(r'\cytokines\compare\Predicted_result.xlsx')
b = pd.read_excel(r'\cytokines\compare\Median_value.xlsx',header = None)

a1 = a['1'].tolist()
b1 = b[0].tolist()

dic1 = {}
for i,j in enumerate(a1):
    dic1[j] = i
    
dic2 = {}
for m,n in enumerate(b1):
    dic2[n] = m


li = []
for i in a['2']:
    if i>0.95:
        li.append(i)
li1 = []
for n,m in enumerate(a['1']):
    if n<len(li):
       li1.append(m)
       
b = pd.read_excel(r'\cytokines\compare\Previous_result.xlsx',header = None)
z1 = [h for h in li1 if h not in list(b[0])]
z2 = [h for h in list(b[0]) if h not in li1]
z3 = [h for h in li1 if h not in z1]
z4 = [h for h in list(b[0]) if h not in z2]
z5 = z1+z2+z3

dic3 = {}
for u,k in dic2.items():
    if u in dic1.keys():
        if u in z5:
            jie = -int(dic1[u]-k)
            dic3[u] = jie