import pandas as pd
import numpy as np

import networkx as nx

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfTransformer

from scipy.sparse import csr_matrix
from pandas.api.types import CategoricalDtype

from sklearn.preprocessing import LabelEncoder

import warnings
# Data Assumption - INCAS
# - 1 Pandas DataFramee

# Index(['annotations', 'dataTags', 'embeddedUrls', 'extraAttributes',
#        'imageUrls', 'segments', 'author', 'contentText', 'geolocation', 'id',
#        'language', 'mediaType', 'mediaTypeAttributes', 'mentionedUsers',
#        'name', 'timePublished', 'title', 'url', 'translatedContentText',
#        'translatedTitle'],
#       dtype='object')

le = LabelEncoder()

def coURL(cum):

    warnings.warn("came in")

    cum.dropna(subset=['author'],inplace=True)
    cum['userid'] = le.fit_transform(cum['author'])


    temp = cum.groupby('url', as_index=False).count()
    cum = cum.loc[cum['url'].isin(temp.loc[temp['userid']>1]['url'].to_list())]

    warnings.warn("grouped")

    cum['value'] = 1
    url = dict(zip(list(cum.url.unique()), list(range(cum.url.unique().shape[0]))))
    cum['url'] = cum['url'].apply(lambda x: url[x]).astype(int)
    del url
    

    # Changing Datatype to string
    cum['userid'] = cum['userid'].astype(str)

    userid = dict(zip(list(cum.userid.astype(str).unique()), list(range(cum.userid.unique().shape[0]))))
    cum['userid'] = cum['userid'].astype(str).apply(lambda x: userid[x]).astype(int)
    
    #cum['userid'] = le.fit_transform(cum['userid'].astype(int))
    
    person_c = pd.CategoricalDtype(sorted(cum.userid.unique()), ordered=True)
    thing_c = pd.CategoricalDtype(sorted(cum.url.unique()), ordered=True)
    
    row = cum.userid.astype(person_c).cat.codes
    col = cum.url.astype(thing_c).cat.codes
    sparse_matrix = csr_matrix((cum["value"], (row, col)), shape=(person_c.categories.size, thing_c.categories.size))
    del row, col, person_c, thing_c

    warnings.warn("calculated sparse matrix")

    #cum = pd.pivot_table(cum,'value', 'userid', 'url', aggfunc='max')
    #cum.fillna(0, inplace = True)
    
    vectorizer = TfidfTransformer()
    tfidf_matrix = vectorizer.fit_transform(sparse_matrix)
    similarities = cosine_similarity(tfidf_matrix, dense_output=False)

    df_adj = pd.DataFrame(similarities.toarray())
    del similarities
    #df_adj.index = list(set(le.inverse_transform(cum['userid'].values)))
    #df_adj.columns = list(set(le.inverse_transform(cum['userid'].values)))
    df_adj.index = userid.keys()
    df_adj.columns = userid.keys()
    
    G = nx.from_pandas_adjacency(df_adj)
    del df_adj
    
    G.remove_nodes_from(list(nx.isolates(G)))

    return G




