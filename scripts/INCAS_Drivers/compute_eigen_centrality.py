import networkx as nx
from scipy.stats import norm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

import warnings

INCAS_DIR = "/scratch1/ashwinba/cache/INCAS/phase_2/similarity_graphs"

def compute_eigen(graph_dir,method):
    try:
        G  = nx.read_gexf(os.path.join(graph_dir))
        G.remove_edges_from(nx.selfloop_edges(G))
        G.remove_nodes_from(list(nx.isolates(G)))
        centrality = nx.eigenvector_centrality(G)
        plot_cdf_curve(list(centrality.values()),method)
        #plot_pdf_curve(list(centrality.values()),method)
        centrality_dict =  {key:{'eigen_centrality':centrality[key]} for key in list(centrality.keys())}
        nx.set_node_attributes(G,centrality_dict)
        nx.write_gexf(G,os.path.join(INCAS_DIR,"{METHOD}_INCAS_TA2_eigen.gexf".format(METHOD=method)))
        warnings.warn("Method {METHOD} completed".format(METHOD=method))
    except Exception as e:
        print(e)
        warnings.warn("Network File {METHOD} does not exists".format(METHOD=method))
        return

def plot_cdf_curve(values,method):
    sns.set(font_scale=1.5, rc={'axes.facecolor':'white', 'figure.facecolor':'white'})
    sns.set_style("whitegrid")
    plt.figure()
    sns.kdeplot(data = values, cumulative = True,linewidth=5)
    percentile = np.percentile(values,90)
    users_count = len(list(filter(lambda x:x>=percentile,values)))
    plt.axvline(x = percentile, color = 'r', linestyle='-',linewidth=2,label="90th pct. : " + str(users_count) + " users")
    warnings.warn("90th percentile : " + str(percentile))
    percentile = np.percentile(values,85)
    warnings.warn("95th perentile : " + str(percentile))
    users_count = len(list(filter(lambda x:x>=percentile,values)))
    plt.axvline(x = percentile, color = 'g', linestyle='-',linewidth=2,label='85th pct. : ' + str(users_count) + " users")
    percentile = np.percentile(values,97)
    warnings.warn("97th perentile : " + str(percentile))
    users_count = len(list(filter(lambda x:x>=percentile,values)))
    plt.axvline(x = percentile, color = 'y', linestyle='-',linewidth=2,label='97th pct. : ' + str(users_count) + " users")
    percentile = np.percentile(values,99.999)
    warnings.warn("97th perentile : " + str(percentile))
    users_count = len(list(filter(lambda x:x>=percentile,values)))
    plt.axvline(x = percentile, color = 'black', linestyle='-',linewidth=2,label='99.999th pct. : ' + str(users_count) + " users")
    plt.ylabel("density")
    plt.xlabel("eigen-centrality")
    plt.title(method)
    plt.legend(loc='best')
    plt.savefig(os.path.join(INCAS_DIR,"{METHOD}_CDF.png".format(METHOD=method)))
    plt.show()
    
def plot_pdf_curve(values,method):
    sns.set(font_scale=1.5, rc={'axes.facecolor':'white', 'figure.facecolor':'white'})
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


graph_root_dir = "/scratch1/ashwinba/cache/INCAS/phase_2"
#graphs = {"coRetweet":"coRetweet_INCAS.gexf","textsimilarity":"textsim_INCAS.gexf","fastretweet":"fastretweet_INCAS.gexf","hashSeq":"hashSeq_INCAS.gexf","fusednetwork":"fusedNetwork.gexf"}
#graphs = {"coRetweet":"coRetweet_INCAS_TA2.gexf"}
#graphs = {"textsim":"textsim_INCAS_T2.gexf"}

#graphs = {"hashseq_min_3":"hashSeq_INCAS_TA2_min_hashtags_3.gexf"}
graphs = {"coURL_min_2":"coURL_INCAS_TA2_1.gexf","coURL_min_3":"coURL_INCAS_TA2_1_min_3.gexf"}
#graphs = {"textSim":"textsim_INCAS_T2_V1.gexf"}
for method,graph_dir in graphs.items():
    compute_eigen(os.path.join(graph_root_dir,graph_dir),method)
    








