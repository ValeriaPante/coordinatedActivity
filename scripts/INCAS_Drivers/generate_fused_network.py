import networkx as nx
import glob
import os

GRAPH_DIR = "/scratch1/ashwinba/cache/INCAS"
def read_graph(graph_dir):
    G = nx.read_gexf(graph_dir)
    G = G.remove_edges_from(list(nx.selfloop_edges(G)))
    return G 

def generate_fused_networks(graphs_root_dir):
    graphs_dir = glob.glob(os.path.join(graphs_root_dir,"*.gexf"))
    graphs = [read_graph(g_dir) for g_dir in graphs_dir]
    fusedGraph = None
    for graph in graphs:
        if fusedGraph is None:
            fusedGraph = graph
        else:
            fusedGraph = nx.intersection(fusedGraph,graph)

    nx.write_gexf(os.path.join(GRAPH_DIR,"fusedNetwork.gexf"))



ROOT_DIR = ""
