## Meta-DHGNN:Method for CRS-related cytokines analysis in CAR-T therapy based on meta-learning directed heterogeneous graph neural network

In this study, we propose Meta-DHGNN, a directed and heterogeneous graph neural network analysis method based on meta-learning. The proposed method integrates both directed and heterogeneous algorithms, while the meta-learning module effectively addresses the issue of limited data availability. This approach enables comprehensive analysis of the cytokine network and accurate prediction of CRS-related cytokines. Firstly, to tackle the challenge posed by small datasets, a pre-training phase is conducted using the meta-learning module. Consequently, the directed algorithm constructs an adjacency matrix that accurately captures potential relationships in a more realistic manner. Ultimately, the heterogeneous algorithm employs meta-photographs and multi-head attention mechanisms to enhance the realism and accuracy of predicting cytokine information associated with positive labels. Our experimental verification on the dataset demonstrates that Meta-DHGNN achieves favorable outcomes.

### Network

![img](https://github.com/wzy38828201/Meta-DHGNN/blob/master/network.png)

### Dataset

The data folder contains ACM、IMDB and cytokines data sets.

### Train

We recommend setting up the runtime environment and run Meta-DHGNN via anaconda. The following steps are required in order to run Meta-DHGNN:

#### 1.download

git clone https://github.com/wzy38828201/Meta-DHGNN.git

#### 2.requirements

```
Python 3.7.6
sklearn 0.22.1
numpy 1.18.5
scipy 1.4.1
pandas 0.24.2
dgl 0.9.1
torch 1.11.0+cu113
torch-geometric 2.0.4
dgl-cuda11.0 0.9.0 py37_0
torch-scatter 2.0.9
torch-sparse 0.6.13
```

#### 3.usage

They mainly consist of two folders: main and cytokines. main is used for the experimental part of the two public data sets ACM and IMDB data sets and the main framework of the Meta-DHGNN model; cytokines are used for the code implementation of the cytokines data sets and experimental part constructed by ourselves

main file:
        data:This folder contains two data sets: ACM and IMDB
        draw:Under the folder is the drawing program
        results:Under the folder are the results data for the ablation experiment part of the model
        save and save0:Both of these are folders that hold the running model
        

        main.py:Run the main program
        model.py and get_adj.py:Body structure of the model (heterogeneous and oriented parts)
        run.py:Used to find the optimal hyperparameters of the model
        utils.py:Adjust whether to choose a directional change

cytokines file:
        The public part is the same as in the main folder
        

        experimentsï¼šIn the folder is the data of the cytokines constructed by oneself, which contains the correlation of the data, the characteristics of the data and the related information such as labels
        
        Adjacency_matrix.py:Generate the adjacency matrix of the data
        Characteristic_matrix.py:Generate a feature matrix of the data
        main-Meta-GNN.py:This is the main program for running the Meta-GNN model

