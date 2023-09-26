import numpy as np
import pandas as pd

'''处理log日志中每个epoch的loss，用于画loss曲线'''
paths = [
    r'G:\图神经网络编程\有向异构\dglhan改实验\结果\原始\细胞因子.log',
    r'G:\图神经网络编程\有向异构\dglhan改实验\结果\仅注意力\细胞因子.log',
    r'G:\图神经网络编程\有向异构\dglhan改实验\结果\仅meta\细胞因子.log',
    r'G:\图神经网络编程\有向异构\dglhan改实验\结果\仅adj\细胞因子.log',
    r'G:\图神经网络编程\有向异构\dglhan改实验\结果\all\细胞因子.log',
         ]
pp = 0
for path in paths:
    prob = []
    # path  = '20220912最新结果/原始/ACM/'
    with open(path,'r',encoding='utf-8') as file:
        lines = file.readlines()
        #p = path.split('/')[0] + '/' + path.split('/')[1] + '/'
        if '细胞因子' in path:
            if pp == 0:
                outfile = open(r'G:\图神经网络编程\有向异构\dglhan改实验\结果\原始\细胞因子loss.txt', 'w+', encoding='utf-8')
            if pp == 1:
                outfile1 = open(r'G:\图神经网络编程\有向异构\dglhan改实验\结果\仅注意力\细胞因子loss.txt', 'w+', encoding='utf-8')
            if pp == 2:
                outfile2 = open(r'G:\图神经网络编程\有向异构\dglhan改实验\结果\仅meta\细胞因子loss.txt', 'w+', encoding='utf-8')
            if pp == 3:
                outfile3 = open(r'G:\图神经网络编程\有向异构\dglhan改实验\结果\仅adj\细胞因子loss.txt', 'w+', encoding='utf-8')
            if pp == 4:
                outfile4 = open(r'G:\图神经网络编程\有向异构\dglhan改实验\结果\all\细胞因子loss.txt', 'w+', encoding='utf-8')
            for d in lines:
                if 'Epoch' in d:
                        if pp == 0:
                            outfile.writelines(d)
                        if pp == 1:
                            outfile1.writelines(d)
                        if pp == 2:
                            outfile2.writelines(d)
                        if pp == 3:
                            outfile3.writelines(d)
                        if pp == 4:
                            outfile4.writelines(d)
            outfile.close()
    pp = pp+1
