import pandas as pd
import numpy as np

def xuhao_mingcheng():
    #生成两个对应文件
    adjlist_path = open(r'G:\元学习\Meta-GNN-master\gcn_fewshot\1769\全部的细胞因子.tsv')
    adjlist_file = pd.read_table(adjlist_path, sep="\t")
    adjlist = adjlist_file[["node1_external_id","node2_external_id","combined_score",'#node1','node2']]

    # 计算蛋白标号和节点序号之间的映射
    all_protein = pd.concat([adjlist["node1_external_id"], adjlist["node2_external_id"]])
    all_protein_name = pd.concat([adjlist['#node1'], adjlist['node2']])
    all_protein_name = all_protein_name.drop_duplicates()
    #print(all_protein_name.values)
    all_protein = all_protein.drop_duplicates()
    protein_map = pd.DataFrame(np.arange(len(all_protein)),index=all_protein,columns=["nodes"])
    protein_map.to_csv(r'G:\元学习\Meta-GNN-master\gcn_fewshot\1769\全部的细胞因子节点映射.csv')
    protein_map_name = pd.DataFrame(all_protein_name.values,columns=["nodes"])
    #print(protein_map)
    #print(protein_map_name)
    #a = protein_map.loc[protein_map_name.values]
    #print(a)
    map_name = np.c_[protein_map,protein_map_name.values]
    pd.DataFrame(map_name).to_csv(r'G:\元学习\Meta-GNN-master\gcn_fewshot\1769\标签分类\序号和名称的对应.csv',index = False,header = None)