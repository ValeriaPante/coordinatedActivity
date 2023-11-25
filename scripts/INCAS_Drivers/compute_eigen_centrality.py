import networkx as nx
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import os

INCAS_DIR = "C:/Users/Ashwin/OneDrive/Desktop/infoOps/coordinatedActivity/cache_files"

def compute_eigen(graph_dir,method):
    G  = nx.read_gexf(os.path.join(graph_dir))
    centrality = nx.eigenvector_centrality(G)
    plot_distribution_curve(list(centrality.values()),method)
    centrality_dict =  {key:{'eigen_centrality':centrality[key]} for key in list(centrality.keys())}
    nx.set_node_attributes(G,centrality_dict)
    nx.write_gexf(G,os.path.join(INCAS_DIR,"{METHOD}_INCAS_0908_eigen.gexf".format(METHOD=method)))

def plot_distribution_curve(values,method):
    print(set(values))
    values.sort()
    print(values[-1])
    norms  = norm.pdf(values)
    print(norms)
    x_1 = np.arange(len(norms))
    print(norms[-1])
    plt.plot(x_1,values)
    plt.ylabel("density")
    plt.xlabel("eigen-centrality")
    #plt.savefig(os.path.join(INCAS_DIR,"{METHOD}.png".format(METHOD=method)))
    plt.show()

graph_dir = "C:/Users/Ashwin/OneDrive/Desktop/infoOps/coordinatedActivity/cache_files/coURL_INCAS.gexf"
compute_eigen(graph_dir,"coURL")







