import networkx as nx
import glob
import os
import pandas as pd

import warnings

GRAPH_DIR = ""
ROOT_DIR = ""

def read_graph(graph_dir,graph_type='gexf'):
    warnings.warn("read "+graph_dir)
    if(graph_type == 'gexf'):
        G = nx.read_gexf(os.path.join(graph_dir))
    else:
        G = nx.read_gml(os.path.join(graph_dir))
        
    warnings.warn(str(len(list(G.nodes))))
    return G

def generate_fused_networks(graphs_root_dir, weighted=False,graph_type='gexf'):
    """
    Merges multiple networks in a single one, where two nodes will be connected if they are in at least one of the input networks.

    Args:
        singleFeatureNets: List of networks to merge, each network must be a networkx.Graph object
        weighted: boolean variable indicating if the merged network should me weighted or not. If True, multiple weights for the same edge are grouped taking the maximum.
    Returns:
        M: merged network
    """

    graphs = []
    graph_types = {'gexf':"*.gexf",'gml':"*.gml.gz"}
    graphs_dir = glob.glob(os.path.join(graphs_root_dir,graph_types[graph_type]))
    singleFeatureNets = [read_graph(g_dir,graph_type) for g_dir in graphs_dir]

    for net in singleFeatureNets:
        if weighted:
            df = pd.DataFrame(net.edges(data='weight'))
        else:
            df = pd.DataFrame(net.edges())
        graphs.append(df)

    temp = pd.concat([df for df in graphs])
    temp = temp.loc[temp[0]!=temp[1]]
    
    if weighted:
        temp.columns = ['source', 'target', 'weight']
        temp = temp.groupby(['source', 'target'], as_index=False).max()
    else:
        temp.columns = ['source', 'target']
        
    temp.dropna(inplace=True)

    if weighted:
        M = nx.from_pandas_edgelist(temp, edge_attr=True)
    else:
        M = nx.from_pandas_edgelist(temp)
    
    nx.write_gexf(M,os.path.join(GRAPH_DIR,"fusedNetwork.gexf"))
    return M

def generate_fused_networks_ashwin(graphs_root_dir):
    graphs_dir = glob.glob(os.path.join(graphs_root_dir,"*.gexf"))
    graphs = [read_graph(g_dir) for g_dir in graphs_dir]
    
    #fusedGraph = nx.disjoint_union_all(graphs)
    fusedGraph = nx.Graph()
    for graph in graphs:
        fusedGraph.add_edges_from(graph.edges())
        fusedGraph.add_nodes_from(graph.nodes())

    # Removing Loops
    #fusedGraph = fusedGraph.remove_edges_from(list(nx.selfloop_edges(fusedGraph)))
    
    warnings.warn(str(len(list(fusedGraph.nodes))))
    nx.write_gexf(fusedGraph,os.path.join(GRAPH_DIR,"fusedNetwork.gexf.gz"))

# Generate fused network
generate_fused_networks(ROOT_DIR,graph_type='gml')
