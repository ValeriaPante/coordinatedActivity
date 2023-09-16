import pandas as pd
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Data assumptions:
#   - 2 Pandas dataframes
#     - control: control dataset -> includes only columns ['user', 'entities', 'id']
#     - treated: information Operation dataset -> includes only columns ['tweetid', 'userid', 'urls']

def coRetweet(control, treated):
    control.dropna(inplace=True)
    
    control['userid'] = control['user'].apply(lambda x: dict(x)['id'])
    control['urls'] = control['entities'].apply(lambda x: dict(x)['urls'])
    control = control[['userid', 'urls']].explode('urls')
    control.dropna(inplace=True)
    control['urls'] = control['urls'].apply(lambda x: str(dict(x)['expanded_url'].replace(',', '.')) if x else np.NaN)
    
    treated['urls'] = treated['urls'].astype(str).replace('[]', '').apply(lambda x: x[1:-1].replace("'", '').split(',') if len(x) != 0 else '')
    treated = treated.loc[treated['urls'] != ''].explode('urls')
    
    cum = pd.concat([control, treated])[['userid', 'urls']].dropna()
    cum.drop_duplicates(inplace=True)

    temp = cum.groupby('urls', as_index=False).count()
    cum = cum.loc[cum['urls'].isin(temp.loc[c['userid']>1]['urls'].to_list())]

    cum['value'] = 1
    cum = pd.pivot_table(cum,'value', 'userid', 'urls', aggfunc='max')
    cum.fillna(0, inplace = True)
    
    vectorizer = TfidfTransformer()
    tfidf_matrix = vectorizer.fit_transform(cum)
    similarities = cosine_similarity(tfidf_matrix)

    df_adj = pd.DataFrame(similarities)
    del similarities
    df_adj.index = cum['userid'].astype(str).to_list()
    df_adj.columns = cum['userid'].astype(str).to_list()
    G = nx.from_pandas_adjacency(df_adj)
    del df_adj, cum
    
    G.remove_nodes_from(list(nx.isolates(G)))

    return G