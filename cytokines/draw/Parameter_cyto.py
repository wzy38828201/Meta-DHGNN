# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 15:31:08 2024

@author: lenovo
"""


import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']

#CYTO
##Learning_rate2
#x = [0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01]
#y0 = [0.6861,0.7732,0.8499,0.8997,0.9120,0.9309,0.9245,0.9309,0.9245,0.9309]#Meta-GCN
#y1 = [0.3261,0.3866,0.5539,0.6865,0.7857,0.9687,0.9749,0.9612,0.9612,0.9612]#HAN
#y2 = [0.9373,0.9373,0.9373,0.8623,0.8621,0.9247,0.9247,0.9182,0.9561,0.9624]#DHAN
#y3 = [0.9499,0.9624,0.9624,0.975,0.975,0.975,0.9812,0.975,0.975,0.9812]#Meta-HAN
#y4 = [0.9875,0.9875,0.9875,0.9875,0.9875,0.9875,0.9875,0.9812,0.975,0.9812]#Meta-DHGNN
##Dropout
#x = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
#y0 = [0.9246,0.9246,0.9246,0.9246,0.9246,0.9246,0.9309,0.9309,0.9309]#Meta-GCN
#y1 = [0.9625,0.9625,0.9687,0.9687,0.9624,0.9687,0.9612,0.9612,0.9687]#HAN
#y2 = [0.9182,0.9499,0.9498,0.9624,0.9749,0.9750,0.9561,0.9749,0.9750]#DHAN
#y3 = [0.9562,0.9624,0.9624,0.9624,0.9562,0.9624,0.975,0.9437,0.9305]#Meta-HAN
#y4 = [0.9624,0.9624,0.9624,0.9624,0.9624,0.9624,0.9624,0.9749,0.9875]#Meta-DHGNN
###Weight_decay2
#x = [0.0002,0.0004,0.0006,0.0008,0.001,0.002,0.004,0.006,0.008,0.01]
#y0 = [0.9487,0.9487,0.9487,0.9487,0.9484,0.9428,0.9245,0.9245,0.9245,0.9245]#Meta-GCN
#y1 = [0.9687,0.9687,0.9687,0.9687,0.9687,0.9562,0.9435,0.9309,0.9245,0.9245]#HAN
#y2 = [0.9687,0.9750,0.9750,0.9624,0.9624,0.9561,0.9372,0.9345,0.9412,0.9412]#DHAN
#y3 = [0.9749,0.9749,0.9624,0.9812,0.9812,0.9812,0.9749,0.9812,0.9812,0.9812]#Meta-HAN
#y4 = [0.9875,0.9812,0.9812,0.9875,0.9875,0.9875,0.9875,0.9875,0.9875,0.9875]#Meta-DHGNN
##Learning_rate
x = [0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01]
y0 = [0.9309,0.8245,0.8245,0.8245,0.9309,0.9309,0.9309,0.9309,0.9309,0.9309]#Meta-GCN
y3 = [0.9624,0.9687,0.9687,0.9687,0.9687,0.9687,0.9687,0.975,0.975,0.975]#Meta-HAN
y4 = [0.9025,0.9333,0.9333,0.9333,0.9562,0.9875,0.9875,0.9875,0.9875,0.9875]#Meta-DHGNN
##Weight_decay
#x = [0.0002,0.0004,0.0006,0.0008,0.001,0.002,0.004,0.006,0.008,0.01]
#y0 = [0.9245,0.9245,0.9245,0.9245,0.9245,0.9245,0.9182,0.9245,0.9245,0.9245]#Meta-GCN
#y3 = [0.9812,0.9812,0.9812,0.9812,0.9812,0.9812,0.9812,0.9812,0.9812,0.9812]#Meta-HAN
#y4 = [0.9875,0.9875,0.9875,0.9875,0.9875,0.9875,0.9812,0.9875,0.9749,0.9812]#Meta-DHGNN
##Step
#x = [2,4,6,8,10,12,14,16,18,20]
#y0 = [0.9487,0.9487,0.9372,0.9372,0.9309,0.9309,0.9309,0.9309,0.9309,0.9245]#Meta-GCN
#y3 = [0.9812,0.9687,0.9812,0.9749,0.9499,0.9749,0.975,0.9624,0.9056,0.8664]#Meta-HAN
#y4 = [0.9875,0.9333,0.9333,0.9333,0.9333,0.9333,0.9333,0.9333,0.9333,0.9333]#Meta-DHGNN
##Train_shot_0
#x = [5,8,10,12,15,18,20,25,30,35]
#y0 = [0.9487,0.9487,0.9487,0.9487,0.9487,0.9487,0.9487,0.9487,0.9487,0.9487]#Meta-GCN
#y3 = [0.9812,0.9812,0.9812,0.9812,0.9812,0.9812,0.9812,0.9812,0.9812,0.9812]#Meta-HAN
#y4 = [0.9875,0.9875,0.9875,0.9875,0.9875,0.9875,0.9875,0.9875,0.9875,0.9875]#Meta-DHGNN

font = {'family': 'Times New Roman', 'weight': 'bold', 'size': 20}
# 绘制折线图
plt.figure(figsize=(10,10))
plt.plot(x, y0, label='Meta-GNN',color='black')
#plt.plot(x, y1, label='dglhan',color='blue')
#plt.plot(x, y2, label='dglhan+adj',color='orange')
plt.plot(x, y3, label='dglhan+meta-learning',color='green')
plt.plot(x, y4, label='Meta-DHGNN',color='red')
# 添加图例
plt.legend(title='',prop=font,loc='lower left')

# 设置标题和标签
plt.title('Cytokines',fontdict=font)
plt.xlabel('Learning_rate',fontdict=font)
plt.ylabel('Model performance',fontdict=font)
plt.xticks(fontproperties = 'Times New Roman', size = 20)
plt.yticks(fontproperties = 'Times New Roman', size = 20)

##标注个别点
#for i in range(len(x)):
#    plt.text(x[i], y0[i], str(y0[i]),fontsize=15,color='black',family='Times New Roman')
##    plt.text(x[i], y1[i], str(y1[i]),fontsize=15,color='blue',family='Times New Roman')
##    plt.text(x[i], y2[i], str(y2[i]),fontsize=15,color='orange',family='Times New Roman')
#    plt.text(x[i], y3[i], str(y3[i]),fontsize=15,color='green',family='Times New Roman')
#    plt.text(x[i], y4[i], str(y4[i]),fontsize=15,color='red',family='Times New Roman')

#添加点标记
plt.scatter(x, y0, c='black', marker='o')
#plt.scatter(x, y1, c='blue', marker='o')
#plt.scatter(x, y2, c='orange', marker='o')
plt.scatter(x, y3, c='green', marker='o')
plt.scatter(x, y4, c='red', marker='o')

##根据不同的模型来调刻度
#plt.xticks([10,15,20,25,30,35,40,45,50,55])
#plt.yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

plt.savefig(r'C:\Users\lenovo\Desktop\CYTO_Learning_rate.png',dpi=300)

# 显示图形
plt.show()