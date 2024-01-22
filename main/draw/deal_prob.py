
import numpy as np
import pandas as pd

'''Process the saved labels.csv and logits.csv files to draw the ROC curve'''
def prob(paths):
    for path in paths:
        prob = []
        with open(path + '\logits.csv', 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for line in lines:
                a = []
                s = line.split(',')
                p0 = float(s[0].split('(')[1])
                p1 = float(s[2].split('(')[1])
                p2 = float(s[4].split('(')[1])
                a.append(p0)
                a.append(p1)
                a.append(p2)
                arr = np.array(a)
                softmax_z = np.exp(arr) / sum(np.exp(arr))
                prob.append(softmax_z)

        pd.DataFrame(prob).to_excel(path + '\prob.xlsx', index=False, header=['0', '1', '2'])
        
def label(paths):
    for path in paths:
        labels = []
        with open(path + '\labels.csv', 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for line in lines:
                la = int(line.replace('\n',''))
                if la == 0:
                    labels.append((1,0,0))
                elif la == 1:
                    labels.append((0,1,0))
                elif la == 2:
                    labels.append((0,0,1))

        pd.DataFrame(np.array(labels)).to_excel(path + '\labels_2.xlsx', index=False,header=False)



paths = [
    #r'G:\图神经网络编程\有向异构\dglhan_all_code20220914\results\original\ACM',
    #r'G:\图神经网络编程\有向异构\dglhan_all_code20220914\results\original\IMDB',
    #r'G:\图神经网络编程\有向异构\dglhan_all_code20220914\results\meta\ACM',
#    r'G:\图神经网络编程\有向异构\dglhan_all_code20220914\results\meta\IMDB',
    #r'G:\图神经网络编程\有向异构\dglhan_all_code20220914\results\adj\ACM',
#   r'G:\图神经网络编程\有向异构\dglhan_all_code20220914\results\adj\IMDB',
    #r'G:\图神经网络编程\有向异构\dglhan_all_code20220914\results\all\ACM',
    r'G:\图神经网络编程\有向异构\dglhan_all_code20220914\results\all\IMDB'
         ]

prob(paths)
label(paths)
