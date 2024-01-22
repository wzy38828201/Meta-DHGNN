'''
According to the parameter definition of main.py, different sizes of parameters are set to find the optimal combination of parameters
'''

import os
import time
import random
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6"
random.seed(20)
np.random.seed(20)
# torch.manual_seed(20)
# torch.cuda.manual_seed(20)
# torch.cuda.manual_seed_all(20)
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True


#learning_rate = [0.01]#0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01
learning_rate2 = [0.009]#0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01
#weight_decay = [0.0002,0.0004,0.0006,0.0008,0.001,0.002,0.004,0.006,0.008,0.01]#0.0002,0.0004,0.0006,0.0008,0.001,0.002,0.004,0.006,0.008,0.01
weight_decay2 = [0.0004]#0.0002,0.0004,0.0006,0.0008,0.001,0.002,0.004,0.006,0.008,0.01
dropout = [0.6]#0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9
#step = [2,4,6,8,10,12,14,16,18,20]#2,4,6,8,10,12,14,16,18,20
#train_shot_0 = [10,15,20,25,30,35,40,45,50,55]#10,15,20,25,30,35,40,45,50,55

File = open(r"\draw.py\parameters.txt", 'a+')
li = []
for i in range(1):  # 运行十次
    for l in learning_rate2:
        for l2 in weight_decay2:
                for d in dropout:
                    File.write("lr2"+'\t'+str(l)+'\t'+"weight_decay2"+'\t'+str(l2)+'\t'+"dropout"+'\t'+str(d)+'\t')
                    File.flush()
                    os.system("python \main.py --lr2 "+" "+str(l)+" " +"--weight_decay2" +" "+str(l2)+" " +"--dropout" +" "+str(d))
                    time.sleep(3)#Preferably 300.
