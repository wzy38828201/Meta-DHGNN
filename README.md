## Usage
=> 1. The data folder contains ACM and IMDB data sets. The imdb data in the IMDB folder is reprocessed based on the original file.

=> 2. ACM run command：`python main.py --dataset ACM`；IMDB run command：`python main.py --dataset IMDB`；

=> 3. The model parameters have been written to the `main.py` function and can be adjusted directly in `main.py`.

=> 4. In addition to individual adjustments in `main.py`, you can also use `run.py`, see `run.py` for details.

## Parameter setting
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
