import networkx as nx
from scipy.stats import norm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

import warnings
import glob

INCAS_DIR = "/scratch1/ashwinba/data/INCAS/EVAL_2B/distributions"

def compute_eigen(graph_dir,method):
    try:
        G  = nx.read_gexf(os.path.join(graph_dir))
        G.remove_edges_from(nx.selfloop_edges(G))
        G.remove_nodes_from(list(nx.isolates(G)))
        centrality = nx.eigenvector_centrality(G)
        plot_cdf_curve(list(centrality.values()),method)
        #plot_pdf_curve(list(centrality.values()),method)
        #centrality_dict =  {key:{'eigen_centrality':centrality[key]} for key in list(centrality.keys())}
        #nx.set_node_attributes(G,centrality_dict)
        #nx.write_gexf(G,os.path.join(INCAS_DIR,"{METHOD}_INCAS_TA2_eigen.gexf".format(METHOD=method)))
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
    percentile = np.percentile(values,95)
    users_count = len(list(filter(lambda x:x>=percentile,values)))
    plt.axvline(x = percentile, color = 'r', linestyle='-',linewidth=4,label="95th pct. : " + str(users_count) + " users")
    percentile = np.percentile(values,90)
    users_count = len(list(filter(lambda x:x>=percentile,values)))
    plt.axvline(x = percentile, color = 'g', linestyle='-',linewidth=2,label='90th pct. : ' + str(users_count) + " users")
    percentile = np.percentile(values,80)
    users_count = len(list(filter(lambda x:x>=percentile,values)))
    plt.axvline(x = percentile, color = 'y', linestyle='-',linewidth=5,label='80th pct. : ' + str(users_count) + " users")
    percentile = np.percentile(values,70)
    warnings.warn("70th perentile : " + str(percentile))
    users_count = len(list(filter(lambda x:x>=percentile,values)))
    plt.axvline(x = percentile, color = 'black', linestyle='-',linewidth=2,label='70th pct. : ' + str(users_count) + " users")
    percentile = np.percentile(values,98)
    users_count = len(list(filter(lambda x:x>=percentile,values)))
    plt.axvline(x = percentile, color = 'blue', linestyle='-',linewidth=2,label='98th pct. : ' + str(users_count) + " users")
    percentile = np.percentile(values,99)
    users_count = len(list(filter(lambda x:x>=percentile,values)))
    plt.axvline(x = percentile, color = 'red', linestyle='-',linewidth=2,label='99th pct. : ' + str(users_count) + " users")
    percentile = np.percentile(values,99.5)
    users_count = len(list(filter(lambda x:x>=percentile,values)))
    plt.axvline(x = percentile, color = 'red', linestyle='-',linewidth=2,label='99.5th pct. : ' + str(users_count) + " users")
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


PATH =  "/scratch1/ashwinba/data/INCAS/EVAL_2B/indicators"

# Fused
PATH =  "/scratch1/ashwinba/data/INCAS/EVAL_2B/indicators/fused"

graphs = glob.glob(os.path.join(PATH,"*.gexf"))

for method in graphs:
    method_name = os.path.basename(method).split("_INCAS")[0]
    compute_eigen(method,method_name)





