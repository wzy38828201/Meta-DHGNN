
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

                a.append(p0)
                a.append(p1)

                arr = np.array(a)
                softmax_z = np.exp(arr) / sum(np.exp(arr))
                prob.append(softmax_z)

        pd.DataFrame(prob).to_excel(path + '\prob.xlsx', index=False, header=['0', '1'])

def label(paths):
    for path in paths:
        labels = []

        with open(path + '\labels.csv', 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for line in lines:
                la = int(line.replace('\n',''))
                if la == 0:
                    labels.append((1,0))
                elif la == 1:
                    labels.append((0,1))
        pd.DataFrame(np.array(labels)).to_excel(path + '\labels_2.xlsx', index=False,header=False)



paths = [
#     r'\results\Meta-GNN',
#     r'\results\original',
#     r'\results\meta',
#    r'\results\adj',
     r'\results\all'
         ]

prob(paths)
label(paths)
