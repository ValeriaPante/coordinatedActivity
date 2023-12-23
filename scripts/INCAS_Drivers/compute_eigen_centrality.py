import networkx as nx
from scipy.stats import norm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

import warnings

INCAS_DIR = "/scratch1/ashwinba/cache/INCAS/similarity_graphs"

def compute_eigen(graph_dir,method):
    try:
        G  = nx.read_gexf(os.path.join(graph_dir))
        centrality = nx.eigenvector_centrality(G)
        #plot_cdf_curve(list(centrality.values()),method)
        plot_pdf_curve(list(centrality.values()),method)
        centrality_dict =  {key:{'eigen_centrality':centrality[key]} for key in list(centrality.keys())}
        nx.set_node_attributes(G,centrality_dict)
        nx.write_gexf(G,os.path.join(INCAS_DIR,"{METHOD}_INCAS_0908_eigen.gexf".format(METHOD=method)))
        warnings.warn("Method {METHOD} completed".format(METHOD=method))
    except:
        warnings.warn("Network File {METHOD} does not exists".format(METHOD=method))
        return


def plot_cdf_curve(values,method):
    sns.set(font_scale=1.5, rc={'axes.facecolor':'white', 'figure.facecolor':'white', 'figure.figsize':(4,4)})
    sns.set_style("whitegrid")
    plt.figure()
    sns.kdeplot(data = values, cumulative = True,linewidth=5)
    plt.title(method)
    plt.savefig(os.path.join(INCAS_DIR,"{METHOD}_CDF.png".format(METHOD=method)))
    plt.show()
    
def plot_pdf_curve(values,method):
    sns.set(font_scale=1.5, rc={'axes.facecolor':'white', 'figure.facecolor':'white', 'figure.figsize':(6,6)})
    sns.set_style("whitegrid")
    plt.figure()
    sns.kdeplot(data = values, cumulative = False,linewidth=5)
    plt.ylabel("density")
    plt.xlabel("eigen-centrality")
    plt.title(method)
    plt.savefig(os.path.join(INCAS_DIR,"{METHOD}_PDF.png".format(METHOD=method)))
    plt.show()

#graph_dir = "/scratch1/ashwinba/cache/INCAS/coRetweet_INCAS.gexf"
#compute_eigen(graph_dir,"coRetweet")


graph_root_dir = "/scratch1/ashwinba/cache/INCAS"
graphs = {"coRetweet":"coRetweet_INCAS.gexcf","textsimilarity":"textsim_INCAS.gexf","fastretweet":"fastretweet_INCAS.gexf","hashSeq":"hashSeq_INCAS.gexf","fusednetwork":"fusedNetwork.gexf"}

for method,graph_dir in graphs.items():
    compute_eigen(os.path.join(graph_root_dir,graph_dir),method)
    








