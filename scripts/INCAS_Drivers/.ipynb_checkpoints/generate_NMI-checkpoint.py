import warnings
import os
import glob

import pandas as pd
import numpy as np

import networkx as nx

from sklearn.metrics import normalized_mutual_info_score

import matplotlib.pyplot as plt
import seaborn as sns


GRAPH_DIR = "/scratch1/ashwinba/cache/INCAS/NMI_INCAS"

def generate_NMI(graphs_dir,threshold):
    nodes = []
    graphs_dicts = {}
    filtered_dict = {}
    for graph in graphs_dir:
        method_name = os.path.basename(graph).split('_')[0]
        G = nx.read_gexf(graph)
        graphs_dicts[method_name] = G
        del G
        nodes+=list(G.nodes)
    
    # Unique Set
    nodes = list(set(list(nodes)))

    # Compute eigen centralities
    for method_name,graph in graphs_dicts.items():
        eigen_centrality = nx.eigenvector_centrality(graph)
        
        percentile = np.percentile(list(eigen_centrality.values()),threshold)
        
        # Filtering nodes
        filtered_nodes = dict(filter(lambda item: item[1]>=percentile,eigen_centrality.items()))

        # Deleting Eigen Centrality Object
        del eigen_centrality

        # filtered_dict 
        filtered_dict[method_name] = list(set(filtered_nodes.keys()))
        
        # Sample Warnings
        warnings.warn(str(method_name)+" done")
        
        # Checking Length
        warnings.warn(str(len(filtered_dict[method_name])) + " totally present")

    # Constructing DataFrame
    for method_name,filtered_nodes in filtered_dict.items():
        series_obj = pd.Series(nodes,index=nodes)
        series_obj = series_obj.map(lambda x: 1 if x in filtered_nodes else 0)
        filtered_dict[method_name] = series_obj
        del series_obj

    final_df = pd.DataFrame(filtered_dict)
    final_df.to_csv(os.path.join(GRAPH_DIR,"NMI_RESULTS_CSV_{THRESH}.csv".format(THRESH=str(threshold))))

    # Calculate NMI
    methods = list(filtered_dict.keys())

    # Calculate NMI
    NMI_dict ={}
    for method in methods:
        NMI_arr = [normalized_mutual_info_score(final_df[method],final_df[method1]) for method1 in methods]
        NMI_dict[method] = pd.Series(NMI_arr,index=methods)


    NMI_df = pd.DataFrame(NMI_dict,columns=methods)
    NMI_df.to_csv(os.path.join(GRAPH_DIR,"NMI_REPORT_{THRESH}.csv".format(THRESH=threshold)))

    
    
    warnings.warn(str(NMI_df.shape))

    # Plot NMI Heatmap
    sns.heatmap(NMI_df)
    plt.savefig(os.path.join(GRAPH_DIR,"NMI_{THRESH}.png".format(THRESH=str(threshold))))

    del NMI_df

        
ROOT_DIR = "/scratch1/ashwinba/cache/INCAS"
graphs_dir = glob.glob(os.path.join(ROOT_DIR,"*.gexf"))
warnings.warn(str(graphs_dir))

generate_NMI(graphs_dir=graphs_dir,threshold=0.95)
    
# Iterative
#threshs = np.arange(0.50,0.98,0.03).tolist()
# for thresh in threshs:
#     generate_NMI(graphs_dir=graphs_dir,threshold=thresh)
