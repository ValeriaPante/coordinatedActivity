import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

import warnings

# Data assumptions:
#   - 2 Pandas dataframes
#     - control: control dataset -> includes only columns ['user', 'entities', 'id']
#     - treated: information Operation dataset -> includes only columns ['tweetid', 'userid', 'urls']

le = LabelEncoder()

def coRetweet(control, treated):

    # Removing NaN Records
    control.dropna(inplace=True)
    treated.dropna(inplace=True)
    
    control['userid'] = control['user'].apply(lambda x: eval(x)['id'])
    control['urls'] = control['entities'].apply(lambda x: eval(x)['urls'])
    control = control[['userid', 'urls']].explode('urls')
    control.dropna(inplace=True)
    
    # Dummy Print
    warnings.warn(str(control['urls'].values))
    
    control['urls'] = control['urls'].apply(lambda x: str(dict(x)['expanded_url']).replace(',', '.') if x else np.NaN)
    
    treated['urls'] = treated['urls'].astype(str).replace('[]', '').apply(lambda x: x[1:-1].replace("'", '').split(',') if len(x) != 0 else '')
    treated = treated.loc[treated['urls'] != ''].explode('urls')
    
    cum = pd.concat([control, treated])[['userid', 'urls']].dropna()
    cum.drop_duplicates(inplace=True)

    temp = cum.groupby('urls', as_index=False).count()
    cum = cum.loc[cum['urls'].isin(temp.loc[temp['userid']>1]['urls'].to_list())]

    cum['value'] = 1
    urls = dict(zip(list(cum.urls.unique()), list(range(cum.urls.unique().shape[0]))))
    cum['urls'] = cum['urls'].apply(lambda x: urls[x]).astype(int)
    del urls
    
    # Changing Datatype to string
    cum['userid'] = cum['userid'].astype(str)

    userid = dict(zip(list(cum.userid.astype(str).unique()), list(range(cum.userid.unique().shape[0]))))
    cum['userid'] = cum['userid'].astype(str).apply(lambda x: userid[x]).astype(int)
    
    #cum['userid'] = le.fit_transform(cum['userid'].astype(int))
    
    person_c = pd.CategoricalDtype(sorted(cum.userid.unique()), ordered=True)
    thing_c = pd.CategoricalDtype(sorted(cum.urls.unique()), ordered=True)
    
    row = cum.userid.astype(person_c).cat.codes
    col = cum.urls.astype(thing_c).cat.codes
    sparse_matrix = csr_matrix((cum["value"], (row, col)), shape=(person_c.categories.size, thing_c.categories.size))
    del row, col, person_c, thing_c

    #cum = pd.pivot_table(cum,'value', 'userid', 'urls', aggfunc='max')
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
