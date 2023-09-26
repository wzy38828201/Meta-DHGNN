import numpy as np
import pandas as pd

'''处理log日志中每个epoch的loss，用于画loss曲线'''
paths = [
     r'G:\图神经网络编程\有向异构\dglhan_all_code20220914\results\原始\ACM\acmloss.txt',
     r'G:\图神经网络编程\有向异构\dglhan_all_code20220914\results\原始\DBLP\dblploss.txt',
     r'G:\图神经网络编程\有向异构\dglhan_all_code20220914\results\原始\IMDB\imdbloss.txt',
     r'G:\图神经网络编程\有向异构\dglhan_all_code20220914\results\仅注意力\ACM\acmloss.txt',
     r'G:\图神经网络编程\有向异构\dglhan_all_code20220914\results\仅注意力\DBLP\dblploss.txt',
     r'G:\图神经网络编程\有向异构\dglhan_all_code20220914\results\仅注意力\IMDB\imdbloss.txt',
     r'G:\图神经网络编程\有向异构\dglhan_all_code20220914\results\仅meta\ACM\acmloss.txt',
     r'G:\图神经网络编程\有向异构\dglhan_all_code20220914\results\仅meta\DBLP\dblploss.txt',
     r'G:\图神经网络编程\有向异构\dglhan_all_code20220914\results\仅meta\IMDB\imdbloss.txt',
     r'G:\图神经网络编程\有向异构\dglhan_all_code20220914\results\仅adj\ACM\acmloss.txt',
     r'G:\图神经网络编程\有向异构\dglhan_all_code20220914\results\仅adj\DBLP\dblploss.txt',
     r'G:\图神经网络编程\有向异构\dglhan_all_code20220914\results\仅adj\IMDB\imdbloss.txt',
     r'G:\图神经网络编程\有向异构\dglhan_all_code20220914\results\all\ACM\acmloss.txt',
     r'G:\图神经网络编程\有向异构\dglhan_all_code20220914\results\all\DBLP\dblploss.txt',
     r'G:\图神经网络编程\有向异构\dglhan_all_code20220914\results\all\IMDB\imdbloss.txt'
         ]
# for path in paths:
#     prob = []
#     # path  = '20220912最新结果/原始/ACM/'
#     with open(path,'r',encoding='utf-8') as file:
#         lines = file.readlines()
#         p = path.split('/')[0] + '/' + path.split('/')[1] + '/'
#         if 'ACM' in path:
#             outfile = open(path, 'w+', encoding='utf-8')
#             for d in lines:
#                 if 'Epoch' in d:
#                         outfile.writelines(d)
#             outfile.close()
#         elif 'DBLP' in path:
#             outfile = open(path + '\DBLPloss.txt', 'w+', encoding='utf-8')
#             for d in lines:
#                 if 'Epoch' in d:
#                         outfile.writelines(d)
#             outfile.close()
#         elif 'IMDB' in path:
#             outfile = open(path + '\IMDBloss.txt', 'w+', encoding='utf-8')
#             for d in lines:
#                 if 'Epoch' in d:
#                         outfile.writelines(d)
#             outfile.close()
for path in paths:
    prob = []
    # path  = '20220912最新结果/原始/ACM/'
    with open(path,'r',encoding='utf-8') as file:
        outfile = open(path, 'w+', encoding='utf-8')
        for d in lines:
            if 'Epoch' in d:
                    outfile.writelines(d)
        outfile.close()