import pandas as pd
import numpy as np
import networkx as nx
from networkx.algorithms import bipartite

# Retweets assumptions:
#   - Pandas dataframe
#   - 'userid' is the user ID of the retweet
#   - 'retweet_tweetid' is the tweet ID of the retweeted tweet
# threshold: edge weight under which edges are discarded 


def co_retweet_network(retweets, threshold = 0.000025):
            
    retweets = retweets.groupby(retweets.columns.tolist(),as_index=False).size()
    retweets.columns = ['userid', 'retweet_tweetid','size']
    filt = retweets[['userid', 'size']].groupby(['userid'],as_index=False).sum()
    filt = list(filt.loc[filt['size'] >= 20]['userid'])
    retweets = retweets.loc[retweets['userid'].isin(filt)]
    graph = nx.from_pandas_edgelist(retweets, 'userid', 'retweet_tweetid','size')

    proj = bipartite.weighted_projected_graph(graph, filt, ratio =True)
    proj.remove_edges_from([(user1, user2) for user1, user2, weight in list(proj.edges(data=True)) if weight['weight'] < threshold])
    
    #returns the co-retweet network
    return proj
