import networkx as nx
import glob
import os

import warnings

GRAPH_DIR = "/scratch1/ashwinba/cache/INCAS"
def read_graph(graph_dir):
    #print(graph_dir)
    warnings.warn("read "+graph_dir)
    G = nx.read_gexf(os.path.join(graph_dir))
    #G = G.remove_edges_from(list(nx.selfloop_edges(G)))
    warnings.warn(str(len(list(G.nodes))))
    return G 

def generate_fused_networks(graphs_root_dir):
    graphs_dir = glob.glob(os.path.join(graphs_root_dir,"*.gexf"))
    graphs = [read_graph(g_dir) for g_dir in graphs_dir]
    
    fusedGraph = nx.disjoint_union_all(graphs)
    # nodes = []    
    # for graph in graphs:
    #     nodes.append(list(graph.nodes))
    #     if fusedGraph is None:
    #         fusedGraph = graph
    #     else:
    #         fusedGraph = nx.intersection(fusedGraph,graph)

    #     warnings.warn("done with indiv graph")
        
    # nodes = set.intersection(*[set(tuple(elem) for elem in sublist) for sublist in nodes])
    # print(len(nodes))
    
    warnings.warn(str(len(list(fusedGraph.nodes))))
    
    
    # Removing Loops
    #fusedGraph = fusedGraph.remove_edges_from(list(nx.selfloop_edges(fusedGraph)))

    nx.write_gexf(fusedGraph,os.path.join(GRAPH_DIR,"fusedNetwork.gexf"))

ROOT_DIR = "/scratch1/ashwinba/cache/INCAS"

# Generate fused network
generate_fused_networks(ROOT_DIR)
