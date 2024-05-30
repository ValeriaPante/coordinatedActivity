import networkx as nx
import pandas as pd
import numpy as np

import os
import glob

import matplotlib.pyplot as plt
import seaborn as sns

import warnings

OUTPUT_DIR = "/scratch1/ashwinba/data/INCAS/EVAL_2B/distributions"
colors = ['green','blue','red','yellow','black','gray']

def plot_distribution(G,method='coretweet',percentiles=[90,99,99.5]):
    G.remove_edges_from(nx.selfloop_edges(G))
    edges = list(G.edges.data())
    edge_weights = list(map(lambda x:float(x[2]['weight']), edges))

    sns.set_style("whitegrid")
    plt.figure()
    sns.kdeplot(data = edge_weights, cumulative = True,linewidth=5)
    
    for inx in range(len(percentiles)):
        percentile_value = np.percentile(edge_weights,percentiles[inx])
        edges_1 = list(filter(lambda x:x[2]['weight']>=percentile_value,edges))
        
        users = [] 
        for edge in edges_1:users+=[edge[0],edge[1]]
        users = set(users)
        
        plt.axvline(x = percentile_value, color = colors[inx%len(colors)], linestyle='-',linewidth=2,label=f'{percentiles[inx]}th pct. , {len(users)} users; {len(edges_1)} edges')
    
    plt.ylabel("density")
    plt.xlabel("similarities")
    plt.title(method)
    
    plt.legend(loc='best')
    plt.savefig(os.path.join(OUTPUT_DIR,f'{method}_EDGE_CDF.png'))

GRAPHS_DIR = "/scratch1/ashwinba/data/INCAS/EVAL_2B/indicators"

# Fused
#GRAPHS_DIR = "/scratch1/ashwinba/data/INCAS/EVAL_2B/indicators/fused"
graphs = glob.glob(os.path.join(GRAPHS_DIR,"*.gexf"))

graphs = [
    '/scratch1/ashwinba/data/INCAS/EVAL_2B/indicators/hashSeq_INCAS_TA2_minhash_4.gexf',
    '/scratch1/ashwinba/data/INCAS/EVAL_2B/indicators/hashSeq_INCAS_TA2_minhash_3.gexf'
    ]

for graph in graphs:
    warnings.warn(f'Processing {graph}')
    G = nx.read_gexf(graph)
    method = os.path.basename(graph).split('.gexf')[0]
    plot_distribution(G,method,percentiles=[1,5,20,40,70,80,90,99,99.5,99.9])
    del G