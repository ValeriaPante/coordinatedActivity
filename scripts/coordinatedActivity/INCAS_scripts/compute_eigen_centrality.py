import networkx as nx
import os
gephi_DIR = "../../../gephi"
graph_DIR = "coRetweet_INCAS.gexf"

G = nx.read_gexf(os.path.join(gephi_DIR,graph_DIR))

centrality = nx.eigenvector_centrality(G)
centrality_dict =  {key:{'eigen_centrality':centrality[key]} for key in list(centrality.keys())}

# Eigen Parameter

nx.set_node_attributes(G,centrality_dict)
nx.write_gexf(G,os.path.join(gephi_DIR,"updated.gexf"))