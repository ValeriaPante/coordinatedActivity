import pandas as pd
import numpy as np
import networkx as nx
from networkx.algorithms import bipartite

# Data assumptions:
#   - Pandas dataframe
#   - 'userId' is the user ID of the retweet
#   - 'url' is the shared URL
# threshold: edge weight under which edges are discarded 


def co_retweet_network(data, threshold = 0.000025):
            
    graph = nx.from_pandas_edgelist(data, 'userId', 'url')
    filt = list(df['userId'].drop_duplicates())

    proj = bipartite.weighted_projected_graph(graph, filt, ratio =True)
    proj.remove_edges_from([(user1, user2) for user1, user2, weight in list(proj.edges(data=True)) if weight['weight'] < threshold])
    
    #returns the co-retweet network
    return proj
