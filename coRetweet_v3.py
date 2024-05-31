import pandas as pd
import numpy as np

import networkx as nx

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfTransformer

from scipy.sparse import csr_matrix
from pandas.api.types import CategoricalDtype
import warnings

# Data assumptions:
#   - 2 Pandas dataframes
#     - control: control dataset -> includes only columns ['user', 'retweeted_status', 'id']
#     - treated: information Operation dataset -> includes only columns ['tweetid', 'userid', 'retweet_id']


# Data Assumption - INCAS
# - 1 Pandas DataFramee

# Index(['annotations', 'dataTags', 'embeddedUrls', 'extraAttributes',
#        'imageUrls', 'segments', 'author', 'contentText', 'geolocation', 'id',
#        'language', 'mediaType', 'mediaTypeAttributes', 'mentionedUsers',
#        'name', 'timePublished', 'title', 'url', 'translatedContentText',
#        'translatedTitle'],
#       dtype='object')

# Mandatory Columns
# ['retweet_id','tweet_id','userid']
def coRetweet(cum):

    cum.rename({"retweet_tweetid":"retweet_id"},axis=1,inplace=True)

    cum.loc[cum["engagementType"] !="retweet", "retweet_id"] = np.nan
    cum.dropna(subset=['retweet_id'],inplace=True)
    
    #cum = cum.rename(index=str,columns={'id':'tweetid'})

    filt = cum[['userid', 'tweetid']].groupby(['userid'],as_index=False).count()
    filt = list(filt.loc[filt['tweetid'] >= 10]['userid'])
    cum = cum.loc[cum['userid'].isin(filt)]
    cum = cum[['userid', 'retweet_id']].drop_duplicates()
    
    del filt

    temp = cum.groupby('retweet_id', as_index=False).count()
    # cum = cum.loc[cum['retweet_id'].isin(temp.loc[temp['userid']>1]['retweet_id'].to_list())]
    cum = cum.loc[cum['retweet_id'].isin(temp.loc[temp['userid']>=10]['retweet_id'].to_list())]

    cum['value'] = 1
    
    ids = dict(zip(list(cum.retweet_id.unique()), list(range(cum.retweet_id.unique().shape[0]))))
    cum['retweet_id'] = cum['retweet_id'].apply(lambda x: ids[x]).astype(int)
    del ids

    userid = dict(zip(list(cum.userid.astype(str).unique()), list(range(cum.userid.unique().shape[0]))))
    cum['userid'] = cum['userid'].astype(str).apply(lambda x: userid[x]).astype(int)
    
    person_c = CategoricalDtype(sorted(cum.userid.unique()), ordered=True)
    thing_c = CategoricalDtype(sorted(cum.retweet_id.unique()), ordered=True)
    
    row = cum.userid.astype(person_c).cat.codes
    col = cum.retweet_id.astype(thing_c).cat.codes
    sparse_matrix = csr_matrix((cum["value"], (row, col)), shape=(person_c.categories.size, thing_c.categories.size))
    del row, col, person_c, thing_c
    
    vectorizer = TfidfTransformer()
    tfidf_matrix = vectorizer.fit_transform(sparse_matrix)
    similarities = cosine_similarity(tfidf_matrix, dense_output=False)

    df_adj = pd.DataFrame(similarities.toarray())
    del similarities
    df_adj.index = userid.keys()
    df_adj.columns = userid.keys()
    G = nx.from_pandas_adjacency(df_adj)
    del df_adj
    
    G.remove_nodes_from(list(nx.isolates(G)))

    return G
