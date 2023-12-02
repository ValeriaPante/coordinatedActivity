import networkx as nx
from scipy.stats import norm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

INCAS_DIR = "/scratch1/ashwinba/cache/INCAS/similarity_graphs"

def compute_eigen(graph_dir,method):
    G  = nx.read_gexf(os.path.join(graph_dir))
    centrality = nx.eigenvector_centrality(G)
    plot_distribution_curve(list(centrality.values()),method)
    centrality_dict =  {key:{'eigen_centrality':centrality[key]} for key in list(centrality.keys())}
    nx.set_node_attributes(G,centrality_dict)
    nx.write_gexf(G,os.path.join(INCAS_DIR,"{METHOD}_INCAS_0908_eigen.gexf".format(METHOD=method)))

def plot_distribution_curve(values,method):
    sns.set(font_scale=1.5, rc={'axes.facecolor':'white', 'figure.facecolor':'white', 'figure.figsize':(4,4)})
    sns.set_style("whitegrid")
    plt.figure()
    sns.kdeplot(data = values, cumulative = True,linewidth=5)
    plt.title(method)
    plt.savefig(os.path.join(INCAS_DIR,"{METHOD}.png".format(METHOD=method)))
    #plt.show()

graph_dir = "/scratch1/ashwinba/cache/INCAS/hashSeq_INCAS.gexf"
compute_eigen(graph_dir,"hashSeq")







