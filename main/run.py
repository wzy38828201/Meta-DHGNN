'''
根据main.py的参数定义，进行参数不同大小的设置，寻找最优参数组合
'''
import os
import time

learning_rate = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008]
learning_rate2 = [0.01,0.001,0.005,0.0001]
dropout = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]

for i in range(1):  # 运行十次
    for l in learning_rate:
        for l2 in learning_rate2:
                for d in dropout:
                        # ubuntu/linux下使用如下命令：
                        # cmd1 = 'nohup python main.py --lr %f ' \
                        #       '--lr2 %f ' \
                        #       '--dropout %f' \
                        #       '> ' \
                        #       'nohup/acm_%d_lr%0.5f_lr2%0.5f_dr%0.2f.log 2>&1 &' % (l,l2,d,i,l,l2,d)
                        # windows下使用如下命令：
                        cmd2 = 'nohup python main.py --lr %f ' \
                              '--lr2 %f ' \
                              '--dropout %f' \
                              '> ' \
                              'log/acm_%d_lr%0.5f_lr2%0.5f_dr%0.2f.log 2>&1 &' % (l, l2, d, i, l, l2, d)

                        os.system(cmd2)
                        time.sleep(300)  # 注意命令之间运行的时间间隔，时间太短系统会同时运行多条命令，会知道CPU/GPU过载，内存溢出等