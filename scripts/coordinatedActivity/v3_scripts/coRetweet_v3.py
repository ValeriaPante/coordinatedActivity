import pandas as pd
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import csr_matrix
from pandas.api.types import CategoricalDtype

import warnings

from sklearn.preprocessing import LabelEncoder

# Data assumptions:
#   - 2 Pandas dataframes
#     - control: control dataset -> includes only columns ['user', 'retweeted_status', 'id']
#     - treated: information Operation dataset -> includes only columns ['tweetid', 'userid', 'retweet_tweetid']

def apply_type_cast(x):
    try:
        return str(int(x))
    except:
        return str(x).strip()

def coRetweet(control, treated):

    # Label Encoder 
    le = LabelEncoder()

    control.dropna(inplace=True)
    treated.dropna(inplace=True)
    
    #control = control.ffill()
    #treated = treated.ffill()
    
    # Old Code
    #control['retweet_id'] = control['retweeted_status'].apply(lambda x: int(dict(x)['id']))
    #control['userid'] = control['user'].apply(lambda x: int(dict(x)['id']))
    
    # New Code
    control['retweet_id'] = control['retweeted_status'].apply(lambda x: int(eval(x)['id']))
    control['userid'] = control['user'].apply(lambda x: int(eval(x)['id']))
    
    #control = control[['id', 'userid', 'retweet_id', 'tweet_timestamp', 'retweet_timestamp']]
    #control.columns = ['tweetid', 'userid', 'retweet_tweetid', 'tweet_timestamp', 'retweet_timestamp']
    
    control = control[['id', 'userid', 'retweet_id']]
    control.columns = ['tweetid', 'userid', 'retweet_tweetid']
    
    control['type'] = "control"
    treated['type'] = "treated"

    treated['retweet_tweetid'] = treated['retweet_tweetid'].astype(int)
    
    cum = pd.concat([treated, control])
    cum.dropna(subset=['userid'],inplace=True)
    
    cum.to_csv("cumulative.csv.gz",compression='gzip')
    
    filt = cum[['userid', 'tweetid']].groupby(['userid'],as_index=False).count()
    filt = list(filt.loc[filt['tweetid'] >= 10]['userid'])
    
    #print(filt)
    cum = cum.loc[cum['userid'].isin(filt)]
    cum = cum[['userid', 'retweet_tweetid']].drop_duplicates()
    
    temp = cum.groupby('retweet_tweetid', as_index=False).count()
    cum = cum.loc[cum['retweet_tweetid'].isin(temp.loc[temp['userid']>1]['retweet_tweetid'].to_list())]

    cum['value'] = 1
    
    ids = dict(zip(list(cum.retweet_tweetid.unique()), list(range(cum.retweet_tweetid.unique().shape[0]))))
    cum['retweet_tweetid'] = cum['retweet_tweetid'].apply(lambda x: ids[x]).astype(int)
    #del urls
    print("CUM",len(set(cum['userid'])))
    
    cum['userid'] = cum['userid'].apply(lambda x:apply_type_cast(x))
    userid = dict(zip(list(cum['userid'].astype(str).unique()), list(range(cum['userid'].unique().shape[0]))))
    warnings.warn(str(userid))
    
    cum['userid'] = cum['userid'].astype(str).apply(lambda x: userid[x])
    #cum['userid'] = le.fit_transform(cum['userid'].astype(str))
    
    
    
    person_c = CategoricalDtype(sorted(cum.userid.unique()), ordered=True)
    thing_c = CategoricalDtype(sorted(cum.retweet_tweetid.unique()), ordered=True)
    
    row = cum.userid.astype(person_c).cat.codes
    col = cum.retweet_tweetid.astype(thing_c).cat.codes
    sparse_matrix = csr_matrix((cum["value"], (row, col)), shape=(person_c.categories.size, thing_c.categories.size))
    del row, col, person_c, thing_c
    
    vectorizer = TfidfTransformer()
    tfidf_matrix = vectorizer.fit_transform(sparse_matrix)
    similarities = cosine_similarity(tfidf_matrix, dense_output=False)

    df_adj = pd.DataFrame(similarities.toarray())
    del similarities
    #uniques = set(list(le.inverse_transform(cum['userid'].values)))
    df_adj.index = userid.keys() 
    df_adj.columns = userid.keys()
    #df_adj.index = uniques
    #df_adj.columns = uniques
    G = nx.from_pandas_adjacency(df_adj)
    del df_adj
    
    G.remove_nodes_from(list(nx.isolates(G)))

    return G
