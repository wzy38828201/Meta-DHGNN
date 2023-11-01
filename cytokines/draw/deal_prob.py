
import numpy as np
import pandas as pd

'''处理保存的labels.csv和logits.csv文件，用来画ROC曲线'''
def prob(paths):
    for path in paths:
        prob = []
        # path  = '20220912最新结果/原始/ACM/'
        with open(path + '\logits.csv', 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for line in lines:
                a = []
                s = line.split(',')
                p0 = float(s[0].split('(')[1])
                p1 = float(s[2].split('(')[1])
                #p2 = float(s[6].split('(')[1])
                # p3 = float(s[6].split('(')[1])  # DBLP
                a.append(p0)
                a.append(p1)
                #a.append(p2)
                # a.append(p3)  # DBLP
                arr = np.array(a)
                softmax_z = np.exp(arr) / sum(np.exp(arr))
                prob.append(softmax_z)

        pd.DataFrame(prob).to_excel(path + '\prob.xlsx', index=False, header=['0', '1'])
        # pd.DataFrame(prob).to_csv(path + 'prob.csv',index=False,header=['0','1','2','3'])  # DBLP

def label(paths):
    for path in paths:
        labels = []
        # path  = '20220912最新结果/原始/ACM/'
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
    # 'cytokines/Results/Meta-GNN',
    # 'cytokines/Results/original',
    # 'cytokines/Results/meta',
    'cytokines/Results/adj',
    # 'cytokines/Results/all'
         ]

prob(paths)
label(paths)
