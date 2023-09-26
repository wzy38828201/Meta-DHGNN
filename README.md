## 用法
=> 1. data文件夹下包含ACM和IMDB数据集，其中imdb文件夹下IMDB的数据是依据原始文件重新处理的。

=> 2. ACM运行命令：`python main.py --dataset ACM`；IMDB运行命令：`python main.py --dataset IMDB`；

=> 3. 模型参数已写到`main.py`函数，可以直接在`main.py`调整参数

=> 4. 调参除了个人在`main.py`挨个调整，还可使用`run.py`调参，详情参见`run.py`。

## 参数设置
`dglhan:`

ACM:`{'lr2': 0.01, 'num_heads': [8], 'hidden_units': 8, 'dropout': 0.7, 'weight_decay2': 0.001, 'num_epochs1': 149, 'patience': 100}`

IMDB:`{'lr2': 0.001, 'num_heads': [8], 'hidden_units': 8, 'dropout': 0.6, 'weight_decay2': 0.001, 'num_epochs1': 117, 'patience': 100}`

`dglhan+meta-learning:`

ACM:
`{'lr': 0.007, 'lr2': 0.007, 'num_heads': [8], 'hidden_units': 8, 'dropout': 0.7, 'weight_decay': 0.003, 'weight_decay2': 0.003, 'num_epochs': 20, 'num_epochs1': 25, 'patience': 100, step = 10, train_shot_0 = 30}`

IMDB:
`{'lr': 0.003, 'lr2': 0.01, 'num_heads': [8], 'hidden_units': 8, 'dropout': 0.6, 'weight_decay': 0.001, 'weight_decay2': 0.0001, 'num_epochs': 50, 'num_epochs1': 99, 'patience': 100}`

`dglhan+adj:`

ACM:
`{'lr2': 0.01, 'num_heads': [8], 'hidden_units': 8, 'dropout': 0.7, 'weight_decay2': 0.001, 'num_epochs1': 108, 'patience': 100}`

IMDB:
`{'lr2': 0.001, 'num_heads': [8], 'hidden_units': 8, 'dropout': 0.6, 'weight_decay2': 0.001, 'num_epochs1': 247, 'patience': 100}`

`dglhan+adj+meta-learning

ACM:
`{'lr': 0.007, 'lr2': 0.01, 'num_heads': [8], 'hidden_units': 8, 'dropout': 0.7, 'weight_decay': 0.003, 'weight_decay2': 0.001, 'num_epochs': 20, 'num_epochs1': 150, 'patience': 100, step = 10, train_shot_0 = 30}`

IMDB:
`{'lr': 0.003, 'lr2': 0.003,'num_heads': [8], 'hidden_units': 8, 'dropout': 0.6, 'weight_decay': 0.001, 'weight_decay2': 0.001, 'num_epochs': 50, 'num_epochs1': 41, 'patience': 100}`
