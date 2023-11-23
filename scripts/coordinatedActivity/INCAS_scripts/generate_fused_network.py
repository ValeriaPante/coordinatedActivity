import networkx as nx
import glob
import os

graph_dir = "/scratch1/ashwinba/cache/INCAS"
def read_graph(graphd_dir):


def generate_fused_networks(graphs_dir):
    # graphs = glob.
    graphs = []
    fusedGraph = None
    for graph in graphs:
        if fusedGraph is None:
            fusedGraph = graph
        else:
            fusedGraph = nx.intersection(fusedGraph,graph)

    nx.write_gexf(os.path.join(graph_dir,"fusedNetwork.gexf"))
