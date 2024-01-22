'''
According to the parameter definition of main.py, different sizes of parameters are set to find the optimal combination of parameters
'''

import os
import time
import random
import numpy as np

random.seed(20)
np.random.seed(20)
# torch.manual_seed(20)
# torch.cuda.manual_seed(20)
# torch.cuda.manual_seed_all(20)
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True


#learning_rate = [0.001]#0.0001,0.0002,0.0003,0.0004,0.0005,0.0006,0.0007,0.0008,0.0009,0.001
learning_rate2 = [0.006]#0.0002,0.0004,0.0006,0.0008,0.001,0.002,0.003,0.004,0.005,0.006
#weight_decay = [0.0002,0.0004,0.0006,0.0008,0.001,0.002,0.004,0.006,0.008,0.01]#0.0002,0.0004,0.0006,0.0008,0.001,0.002,0.004,0.006,0.008,0.01
weight_decay2 = [0.0002,0.0004,0.0006,0.0008,0.001,0.002,0.004,0.006,0.008,0.01]#0.0002,0.0004,0.0006,0.0008,0.001,0.002,0.004,0.006,0.008,0.01
dropout = [0.6]#0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9
#step = [2,4,6,8,10,12,14,16,18,20]#2,4,6,8,10,12,14,16,18,20
#train_shot_0 = [5,8,10,12,15,18,20,25,30,35]#10,15,20,25,30,35,40,45,50,55

File = open(r"\draw\parameters.txt", 'a+')
li = []
for i in range(1):  # 
    for l in learning_rate2:
        for l2 in dropout:
                for d in weight_decay2:

                    File.write("lr2"+'\t'+str(l)+'\t'+"dropout"+'\t'+str(l2)+'\t'+"weight_decay2"+'\t'+str(d)+'\t')
                    File.flush()
                    os.system("python \main-Meta-GNN.py --lr2 "+" "+str(l)+" " +"--dropout" +" "+str(l2)+" " +"--weight_decay2" +" "+str(d))
                    time.sleep(3)#
