import pandas as pd
import numpy as np

import networkx as nx

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfTransformer

from scipy.sparse import csr_matrix

import warnings
# Data Assumption - INCAS
# - 1 Pandas DataFramee

# Index(['annotations', 'dataTags', 'embeddedUrls', 'extraAttributes',
#        'imageUrls', 'segments', 'author', 'contentText', 'geolocation', 'id',
#        'language', 'mediaType', 'mediaTypeAttributes', 'mentionedUsers',
#        'name', 'timePublished', 'title', 'url', 'translatedContentText',
#        'translatedTitle'],
#       dtype='object')
# ['userid' --> numerical encoding of author]

def coURL(cum):

    warnings.warn("came in")
    
    # Renaming columns if necessary
    cum.rename({'urls':'embeddedUrls'},axis=1,inplace=True)
    
    cum.dropna(subset=['embeddedUrls'],inplace=True)
    
    cum['urls'] = cum['embeddedUrls'].astype(str).replace('[]', '').apply(lambda x: x[1:-1].replace("'", '').split(',') if len(x) != 0 else '')
    cum = cum.loc[cum['urls'] != ''].explode('urls')
   
    cum.drop_duplicates(subset=['userid'],inplace=True)
    cum = cum[['userid','urls']].dropna()

    temp = cum.groupby('urls', as_index=False).count()

    cum = cum.loc[cum['urls'].isin(temp.loc[temp['userid']>10]['urls'].to_list())]

    cum['value'] = 1
    urls = dict(zip(list(cum.urls.unique()), list(range(cum.urls.unique().shape[0]))))
    cum['urls'] = cum['urls'].apply(lambda x: urls[x]).astype(int)
    del urls
    warnings.warn("initiated urls")

    # Changing Datatype to string
    cum['userid'] = cum['userid'].astype(str)

    userid = dict(zip(list(cum.userid.astype(str).unique()), list(range(cum.userid.unique().shape[0]))))
    cum['userid'] = cum['userid'].astype(str).apply(lambda x: userid[x]).astype(int)
    
    warnings.warn("before categorical")
    
    person_c = pd.CategoricalDtype(sorted(cum.userid.unique()), ordered=True)
    thing_c = pd.CategoricalDtype(sorted(cum.urls.unique()), ordered=True)
    
    row = cum.userid.astype(person_c).cat.codes
    col = cum.urls.astype(thing_c).cat.codes
    sparse_matrix = csr_matrix((cum["value"], (row, col)), shape=(person_c.categories.size, thing_c.categories.size))
    
    warnings.warn("written temp file")
    
    del row, col, person_c, thing_c

    warnings.warn("calculated sparse matrix")
    #print(sparse_matrix)

    #cum = pd.pivot_table(cum,'value', 'userid', 'url', aggfunc='max')
    #cum.fillna(0, inplace = True)
    
    vectorizer = TfidfTransformer()
    tfidf_matrix = vectorizer.fit_transform(sparse_matrix)
    similarities = cosine_similarity(tfidf_matrix, dense_output=False)
    
    print(type(similarities))
    warnings.warn("similarities_detected")

    df_adj = pd.DataFrame(similarities.toarray())
    
    warnings.warn("Calculated df adj")

    del similarities
    #df_adj.index = list(set(le.inverse_transform(cum['userid'].values)))
    #df_adj.columns = list(set(le.inverse_transform(cum['userid'].values)))
    df_adj.index = userid.keys()
    df_adj.columns = userid.keys()
    
    G = nx.from_pandas_adjacency(df_adj)
    del df_adj
    
    warnings.warn("constructed adj matrix")
    
    G.remove_nodes_from(list(nx.isolates(G)))
    G.remove_edges_from(nx.selfloop_edges(G))

    return G



