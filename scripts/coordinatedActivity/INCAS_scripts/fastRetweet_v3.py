import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime
from datetime import timedelta

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix


import warnings

# retrieves tweet's timestamp from its ID
def get_tweet_timestamp(tid):
    try:
        offset = 1288834974657
        tstamp = (tid >> 22) + offset
        utcdttime = datetime.utcfromtimestamp(tstamp/1000)
        return utcdttime
    except:
        return None   
def get_retweet_userid(retweet_id,cum_df):
    if(retweet_id != ''):
        mappings  = list(cum_df.loc[cum_df["tweetid"] == retweet_id]['userid'].values)
        if(len(mappings)!=0):
            return mappings[0]
        else:
            return np.nan
    return ""

# Index(['annotations', 'dataTags', 'embeddedUrls', 'extraAttributes',
#        'imageUrls', 'segments', 'author', 'contentText', 'geolocation', 'id',
#        'language', 'mediaType', 'mediaTypeAttributes', 'mentionedUsers',
#        'name', 'timePublished', 'title', 'url', 'translatedContentText',
#        'translatedTitle'],
#       dtype='object'

def fastRetweet(cum1, timeInterval = 10):
    cum1.dropna(inplace=True)

    cum1['tweet_timestamp'] = cum1['tweetid'].apply(lambda x: get_tweet_timestamp(int(x)))
    cum1['retweet_timestamp'] = cum1['retweet_id'].apply(lambda x: get_tweet_timestamp(int(x)))
    # Calculating retweet_userid
    cum1['retweet_userid'] = cum1['retweet_id'].apply(lambda x:get_retweet_userid(x,cum1))
    cum1 = cum1[['id', 'userid', 'retweet_id', 'tweet_timestamp', 'retweet_timestamp', 'retweet_userid']]
    cum1.columns = ['tweetid', 'userid', 'retweet_tweetid', 'tweet_timestamp', 'retweet_timestamp', 'retweet_userid']
    
    #print("tweet",len(cum["tweet_timestamp"].values))
  
    cum1['delta'] = (cum1['tweet_timestamp'] - cum1['retweet_timestamp']).dt.seconds

    cumulative = cum1[['userid','retweet_userid', 'delta']].copy()
    cumulative['userid'].astype(int).astype(str)
    cumulative = cumulative.loc[cumulative['delta'] <= timeInterval]
    
    cumulative = cumulative.groupby(['userid', 'retweet_userid'],as_index=False).count()
    cumulative = cumulative.loc[cumulative['delta'] > 1]
    
    #cum = nx.from_pandas_edgelist(cumulative, 'userid', 'retweet_userid','delta')

    cum['userid'].astype(int).astype(str)
    cum = cum.loc[cum['delta'] > 1]
    
    urls = dict(zip(list(cum.retweet_userid.unique()), list(range(cum.retweet_userid.unique().shape[0]))))
    cum['retweet_userid'] = cum['retweet_userid'].apply(lambda x: urls[x]).astype(int)
    del urls
    
    userid = dict(zip(list(cum.userid.astype(str).unique()), list(range(cum.userid.unique().shape[0]))))
    cum['userid'] = cum['userid'].astype(str).apply(lambda x: userid[x]).astype(int)
    
    person_c = pd.CategoricalDtype(sorted(cum.userid.unique()), ordered=True)
    thing_c = pd.CategoricalDtype(sorted(cum.retweet_userid.unique()), ordered=True)
    
    row = cum.userid.astype(person_c).cat.codes
    col = cum.retweet_userid.astype(thing_c).cat.codes
    sparse_matrix = csr_matrix((cum["delta"], (row, col)), shape=(person_c.categories.size, thing_c.categories.size))
    del row, col, person_c, thing_c
    
    #cum = pd.pivot_table(cum,'value', 'userid', 'urls', aggfunc='max')
    #cum.fillna(0, inplace = True)
    
    vectorizer = TfidfTransformer()
    tfidf_matrix = vectorizer.fit_transform(sparse_matrix)
    similarities = cosine_similarity(tfidf_matrix, dense_output=False)
    
    
    df_adj = pd.DataFrame(similarities.toarray())
    del similarities
    df_adj.index = userid.keys()
    df_adj.columns = userid.keys()
    G = nx.from_pandas_adjacency(df_adj)
    del df_adj
    
    G.remove_edges_from(nx.selfloop_edges(G))
    G.remove_nodes_from(list(nx.isolates(G)))
    
    warnings.warn(str(len((set(G.nodes)))))

    return G