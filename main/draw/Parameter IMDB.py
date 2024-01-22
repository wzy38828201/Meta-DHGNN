# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 15:31:08 2024

@author: lenovo
"""


import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']

#IMDB
##Learning_rate2
x = [0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01]
y1 = [0.5512,0.5503,0.5493,0.5543,0.5455,0.5472,0.5417,0.5504,0.5533,0.5462]#HAN
y2 = [0.5571,0.5601,0.5537,0.5509,0.5588,0.5548,0.5601,0.5503,0.5606,0.555]#DHAN
y3 = [0.5278,0.5318,0.5254,0.5325,0.5335,0.5359,0.5352,0.5136,0.5088,0.5098]#Meta-HAN
y4 = [0.5462,0.5620,0.5587,0.5660,0.5575,0.5664,0.5558,0.5656,0.5451,0.5571]#Meta-DHGNN
##Dropout
#x = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
#y1 = [0.5473,0.5485,0.5476,0.54,0.5482,0.5543,0.5382,0.5166,0.549]#HAN
#y2 = [0.4984,0.5506,0.5573,0.5587,0.5554,0.5039,0.5573,0.555,0.5157]#DHAN
#y3 = [0.511,0.5164,0.5175,0.543,0.5358,0.5183,0.5369,0.5484,0.5525]#Meta-HAN
#y4 = [0.4851,0.4847,0.4910,0.4980,0.5082,0.5192,0.5284,0.5439,0.5547]#Meta-DHGNN
##Weight_decay2
#x = [0.0002,0.0004,0.0006,0.0008,0.001,0.002,0.004,0.006,0.008,0.01]
#y1 = [0.5265,0.5094,0.5359,0.5323,0.5331,0.5188,0.5329,0.5445,0.5482,0.5543]#HAN
#y2 = [0.5325,0.5261,0.5328,0.527,0.5268,0.5272,0.557,0.5593,0.5598,0.5606]#DHAN
#y3 = [0.5556,0.5564,0.5561,0.5572,0.5595,0.5634,0.5595,0.5378,0.5558,0.5496]#Meta-HAN
#y4 = [0.565,0.5636,0.5579,0.5631,0.5664,0.565,0.5522,0.5343,0.5246,0.513]#Meta-DHGNN
##Learning_rate
#x = [0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01]
#y3 = [0.4947,0.4907,0.5143,0.5206,0.5053,0.5075,0.5015,0.5137,0.5086,0.5335]#Meta-HAN
#y4 = [0.5358,0.5571,0.5591,0.552,0.5476,0.5558,0.5506,0.5473,0.5421,0.5547]#Meta-DHGNN
#Weight_decay
#x = [0.0002,0.0004,0.0006,0.0008,0.001,0.002,0.004,0.006,0.008,0.01]
#y3 = [0.5525,0.5562,0.549,0.5533,0.5524,0.5516,0.5634,0.5597,0.5499,0.5521]#Meta-HAN
#y4 = [0.5684,0.5596,0.5658,0.5616,0.5664,0.5683,0.5661,0.5609,0.5610,0.5495]#Meta-DHGNN
##Step
#x = [2,4,6,8,10,12,14,16,18,20]
#y3 = [0.5634,0.5428,0.5408,0.5484,0.5467,0.5493,0.5574,0.5435,0.5482,0.5434]#Meta-HAN
#y4 = [0.5664,0.5407,0.5453,0.5544,0.5511,0.5440,0.5465,0.5340,0.5258,0.5325]#Meta-DHGNN
##Train_shot_0
#x = [10,15,20,25,30,35,40,45,50,55]
#y3 = [0.5477,0.5611,0.5535,0.5403,0.5634,0.5493,0.5591,0.5534,0.5496,0.5601]#Meta-HAN
#y4 = [0.5516,0.5450,0.5625,0.5618,0.5664,0.5554,0.5581,0.5565,0.5650,0.5624]#Meta-DHGNN

font = {'family': 'Times New Roman', 'weight': 'bold', 'size': 30}
# 绘制折线图
plt.figure(figsize=(10,10))
plt.plot(x, y1, label='dglhan')
plt.plot(x, y2, label='dglhan+adj')
plt.plot(x, y3, label='dglhan+meta-learning')
plt.plot(x, y4, label='Meta-DHGNN')
# 添加图例
plt.legend(title='',prop=font,loc='lower left')

# 设置标题和标签
plt.title('IMDB',fontdict=font)
plt.xlabel('Learning_rate2',fontdict=font)
plt.ylabel('Model performance',fontdict=font)
plt.xticks(fontproperties = 'Times New Roman', size = 30)
plt.yticks(fontproperties = 'Times New Roman', size = 30)

#标注个别点
#for i in range(len(x)):
#    plt.text(x[i], y1[i], str(y1[i]),fontsize=15,color='blue',family='Times New Roman')
#    plt.text(x[i], y2[i], str(y2[i]),fontsize=15,color='orange',family='Times New Roman')
#    plt.text(x[i], y3[i], str(y3[i]),fontsize=15,color='green',family='Times New Roman')
#    plt.text(x[i], y4[i], str(y4[i]),fontsize=15,color='red',family='Times New Roman')

#添加点标记
plt.scatter(x, y1, c='blue', marker='o')
plt.scatter(x, y2, c='orange', marker='o')
plt.scatter(x, y3, c='green', marker='o')
plt.scatter(x, y4, c='red', marker='o')

##根据不同的模型来调刻度
#plt.xticks([10,15,20,25,30,35,40,45,50,55])
#plt.yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

plt.savefig(r'C:\Users\lenovo\Desktop\IMDB_Learning_rate2.png',dpi=300)

# 显示图形
plt.show()