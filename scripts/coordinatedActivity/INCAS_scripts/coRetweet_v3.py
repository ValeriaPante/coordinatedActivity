import pandas as pd
import numpy as np

import networkx as nx

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfTransformer

from scipy.sparse import csr_matrix
from pandas.api.types import CategoricalDtype

from sklearn.preprocessing import LabelEncoder

import warnings

# Data assumptions:
#   - 2 Pandas dataframes
#     - control: control dataset -> includes only columns ['user', 'retweeted_status', 'id']
#     - treated: information Operation dataset -> includes only columns ['tweetid', 'userid', 'retweet_tweetid']


# Data Assumption - INCAS
# - 1 Pandas DataFramee

# Index(['annotations', 'dataTags', 'embeddedUrls', 'extraAttributes',
#        'imageUrls', 'segments', 'author', 'contentText', 'geolocation', 'id',
#        'language', 'mediaType', 'mediaTypeAttributes', 'mentionedUsers',
#        'name', 'timePublished', 'title', 'url', 'translatedContentText',
#        'translatedTitle'],
#       dtype='object')


def coRetweet(cum):

    warnings.warn("came in")

    # Setting up LabelEncoder instance
    le = LabelEncoder()

    # Sorting cum based on timePublished
    cum.sort_values(by=['timePublished'], inplace=True)
    
    warnings.warn("No of Records "+str(len(cum)))

    # Userid
    cum['userid'] = le.fit_transform(cum['author'])
    warnings.warn("after label encoding")

    # FFill for NaN values
    cum['retweet_id'] = np.nan

    contextsCounts = cum['contentText'].value_counts()
    repeatedContexts = contextsCounts.where(contextsCounts > 1).index
    
    # Display no of repeatedCounts
    warnings.warn("repeatedCounts: "+str(len(repeatedContexts)))

    for context in repeatedContexts:
        orginal_id = cum[cum['contentText'] == context]['id'].values[0]
        cum.loc[(cum['contentText'] == context), "retweet_id"] = orginal_id

    warnings.warn("grouped")


    filt = cum[['userid', 'tweetid']].groupby(
        ['userid'], as_index=False).count()

    filt = list(filt.loc[filt['tweetid'] >= 10]['userid'])
    # print(filt)
    cum = cum.loc[cum['userid'].isin(filt)]
    cum = cum[['userid', 'retweet_id']].drop_duplicates()

    temp = cum.groupby('retweet_id', as_index=False).count()
    cum = cum.loc[cum['retweet_id'].isin(
        temp.loc[temp['userid'] > 1]['retweet_id'].to_list())]

    cum['value'] = 1

    ids = dict(zip(list(cum.retweet_tweetid.unique()), list(
        range(cum.retweet_tweetid.unique().shape[0]))))
    cum['retweet_id'] = cum['retweet_id'].apply(
        lambda x: ids[x]).astype(int)
    # del urls
    warnings.warn("CUM "+ str(len(set(cum['userid']))))

    person_c = CategoricalDtype(sorted(cum.userid.unique()), ordered=True)
    thing_c = CategoricalDtype(sorted(cum.retweet_tweetid.unique()), ordered=True)

    row = cum.userid.astype(person_c).cat.codes
    col = cum.retweet_tweetid.astype(thing_c).cat.codes
    sparse_matrix = csr_matrix((cum["value"], (row, col)), shape=(
        person_c.categories.size, thing_c.categories.size))
    del row, col, person_c, thing_c

    vectorizer = TfidfTransformer()
    tfidf_matrix = vectorizer.fit_transform(sparse_matrix)
    similarities = cosine_similarity(tfidf_matrix, dense_output=False)

    df_adj = pd.DataFrame(similarities.toarray())
    del similarities
    uniques = set(list(le.inverse_transform(cum['userid'].values)))
    # df_adj.index = userid.keys()
    # df_adj.columns = userid.keys()
    df_adj.index = uniques
    df_adj.columns = uniques
    G = nx.from_pandas_adjacency(df_adj)
    del df_adj

    G.remove_nodes_from(list(nx.isolates(G)))

    return G
