# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 15:31:08 2024

@author: lenovo
"""


import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']

#ACM
##Learning_rate2
#x = [0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01]
#y1 = [0.8634,0.8619,0.8628,0.8648,0.8612,0.8636,0.8296,0.8615,0.8284,0.8476]#HAN
#y2 = [0.8596,0.8426,0.8246,0.8632,0.8554,0.8527,0.8593,0.8606,0.8644,0.8513]#DHAN
#y3 = [0.8824,0.8755,0.8747,0.8795,0.8799,0.8769,0.8625,0.8772,0.8812,0.8901]#Meta-HAN
#y4 = [0.8661,0.8713,0.8554,0.8654,0.8689,0.882,0.8839,0.8915,0.8737,0.8832]#Meta-DHGNN
##Dropout
#x = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
#y1 = [0.838,0.8382,0.8674,0.8817,0.8842,0.8802,0.8648,0.8453,0.6051]#HAN
#y2 = [0.845,0.8511,0.855,0.8405,0.8658,0.8866,0.8864,0.865,0.8701]#DHAN
#y3 = [0.8502,0.8503,0.8526,0.8710,0.8823,0.8887,0.8717,0.8357,0.7785]#Meta-HAN
#y4 = [0.8673,0.886,0.854,0.8559,0.8659,0.8597,0.8915,0.8709,0.8312]#Meta-DHGNN
##Weight_decay2
#x = [0.0002,0.0004,0.0006,0.0008,0.001,0.002,0.004,0.006,0.008,0.01]
#y1 = [0.8716,0.8722,0.8788,0.8635,0.8842,0.8803,0.8708,0.8693,0.8636,0.8643]#HAN
#y2 = [0.8726,0.8866,0.8641,0.8674,0.8662,0.8604,0.8639,0.8673,0.8644,0.8632]#DHAN
#y3 = [0.8857,0.8901,0.8901,0.8904,0.8856,0.8919,0.8838,0.8815,0.8899,0.8877]#Meta-HAN
#y4 = [0.8866,0.8877,0.8877,0.8731,0.8915,0.8878,0.8816,0.8773,0.8722,0.8761]#Meta-DHGNN
##Learning_rate
#x = [0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01]
#y3 = [0.8801,0.8761,0.866,0.8711,0.8584,0.8647,0.8589,0.868,0.863,0.8709]#Meta-HAN
#y4 = [0.864,0.867,0.8482,0.8395,0.8641,0.8576,0.8618,0.8575,0.8708,0.8832]#Meta-DHGNN
##Weight_decay
#x = [0.0002,0.0004,0.0006,0.0008,0.001,0.002,0.004,0.006,0.008,0.01]
#y3 = [0.8914,0.8649,0.8878,0.8793,0.8805,0.8810,0.8861,0.8784,0.8893,0.8724]#Meta-HAN
#y4 = [0.8569,0.8577,0.8695,0.8632,0.8915,0.8592,0.8584,0.8867,0.8637,0.8722]#Meta-DHGNN
##Step
#x = [2,4,6,8,10,12,14,16,18,20]
#y3 = [0.8919,0.8805,0.8791,0.8866,0.8909,0.8869,0.8764,0.8747,0.8681,0.8758]#Meta-HAN
#y4 = [0.8964,0.8843,0.8934,0.8737,0.8915,0.8810,0.8591,0.8806,0.8547,0.8607]#Meta-DHGNN
##Train_shot_0
x = [10,15,20,25,30,35,40,45,50,55]
y3 = [0.856,0.8887,0.8845,0.875,0.8919,0.887,0.8753,0.8673,0.8674,0.8708]#Meta-HAN
y4 = [0.883,0.8403,0.8512,0.8724,0.8964,0.8950,0.8672,0.8663,0.8758,0.8551]#Meta-DHGNN

font = {'family': 'Times New Roman', 'weight': 'bold', 'size': 20}
# 绘制折线图
plt.figure(figsize=(10,10))
#plt.plot(x, y1, label='dglhan')
#plt.plot(x, y2, label='dglhan+adj')
plt.plot(x, y3, label='dglhan+meta-learning')
plt.plot(x, y4, label='Meta-DHGNN')
# 添加图例
plt.legend(title='',prop=font,loc='lower left')

# 设置标题和标签
plt.title('ACM',fontdict=font)
plt.xlabel('Train_shot_0',fontdict=font)
plt.ylabel('Model performance',fontdict=font)
plt.xticks(fontproperties = 'Times New Roman', size = 20)
plt.yticks(fontproperties = 'Times New Roman', size = 20)

##标注个别点
#for i in range(len(x)):
#    plt.text(x[i], y1[i], str(y1[i]),fontsize=15,color='blue',family='Times New Roman')
#    plt.text(x[i], y2[i], str(y2[i]),fontsize=15,color='orange',family='Times New Roman')
#    plt.text(x[i], y3[i], str(y3[i]),fontsize=15,color='green',family='Times New Roman')
#    plt.text(x[i], y4[i], str(y4[i]),fontsize=15,color='red',family='Times New Roman')

#添加点标记
#plt.scatter(x, y1, c='blue', marker='o')
#plt.scatter(x, y2, c='orange', marker='o')
plt.scatter(x, y3, c='green', marker='o')
plt.scatter(x, y4, c='red', marker='o')

##根据不同的模型来调刻度
#plt.xticks([10,15,20,25,30,35,40,45,50,55])
#plt.yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

plt.savefig(r'C:\Users\lenovo\Desktop\ACM_Train_shot_0.png',dpi=300)

# 显示图形
plt.show()