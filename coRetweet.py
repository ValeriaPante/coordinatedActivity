import pandas as pd
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Data assumptions:
#   - 2 Pandas dataframes
#     - control: control dataset -> includes only columns ['user', 'retweeted_status', 'id']
#     - treated: information Operation dataset -> includes only columns ['tweetid', 'userid', 'retweet_tweetid']

def coRetweet(control, treated):
    control.dropna(inplace=True)
    treated.dropna(inplace=True)
    
    control['retweet_id'] = control['retweeted_status'].apply(lambda x: int(dict(x)['id']))
    control['userid'] = control['user'].apply(lambda x: int(dict(x)['id']))
    control = control[['id', 'userid', 'retweet_id', 'tweet_timestamp', 'retweet_timestamp']]
    control.columns = ['tweetid', 'userid', 'retweet_tweetid', 'tweet_timestamp', 'retweet_timestamp']
    
    treated['retweet_tweetid'] = treated['retweet_tweetid'].astype(int)
    
    cum = pd.concat([treated, control])
    filt = cum[['userid', 'tweetid']].groupby(['userid'],as_index=False).count()
    filt = list(filt.loc[filt['tweetid'] >= 20]['userid'])
    cum = cum.loc[cum['userid'].isin(filt)]
    cum = cum[['userid', 'retweet_tweetid', 'tweetid']].groupby(['userid', 'retweet_tweetid'],as_index=False).size()
    cum.columns = ['userid', 'retweet_tweetid','size']
    
    retweetVectors = pd.DataFrame(cum['userid'].drop_duplicates().to_list(), columns=['userid'])
    retweetVectors['retweets'] = retweetVectors['userid'].apply(lambda x: ' '.join(cum.loc[cum['userid']==x]['retweet_tweetid'].astype(str).to_list()))
    del cum
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(retweetVectors['retweets'])
    similarities = cosine_similarity(tfidf_matrix)

    df_adj = pd.DataFrame(similarities)
    del similarities
    df_adj.index = retweetVectors['userid'].astype(str).to_list()
    df_adj.columns = retweetVectors['userid'].astype(str).to_list()
    G = nx.from_pandas_adjacency(df_adj)
    del df_adj, retweetVectors
    
    G.remove_nodes_from(list(nx.isolates(G)))

    return G