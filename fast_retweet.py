import pandas as pd
import numpy as np
import networkx as nx
from networkx.algorithms import bipartite
import gzip
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from os import listdir
from os.path import isfile, join
from datetime import datetime
from datetime import timedelta
import unfurl
import sys



def get_tweet_timestamp(tid):
    try:
        offset = 1288834974657
        tstamp = (tid >> 22) + offset
        utcdttime = datetime.utcfromtimestamp(tstamp/1000)
        return utcdttime
    except:
        return None
    
# Data assumptions:
#   - Pandas dataframe
#   - 'userid' is the user ID of the retweet
#   - 'tweetid' is the tweet ID of the retweet
#   - 'retweet_tweetid' is the tweet ID of the retweeted tweet
#   - 'retweet_userid' is the user ID of the creator of the retweetd tweet
# timeInterval: time distance between retweet and original tweet under which a retweet is considered fast
# minRetweets: minimum number of fast retweets per user
    
def fast_retweet(data, timeInterval, minRetweets = 1):

    data['retweet_tweetid'] = data['retweet_tweetid'].astype(int)
    data['original_ts'] = data['retweet_tweetid'].apply(lambda x: get_tweet_timestamp(x))
    data['tweetid'] = data['tweetid'].astype(int)
    data['tweetTime'] = data['tweetid'].apply(lambda x: get_tweet_timestamp(x))
    data['delta'] = (data['tweetTime'] - data['original_ts']).dt.seconds
    
    data = data.loc[data['delta'] <= timeInterval]
    
    data = data.groupby(['userid', 'retweet_userid'],as_index=False).count()
    data = data.loc[data['delta'] > minRetweets]
    
    graph = nx.from_pandas_edgelist(data, 'userid', 'retweet_userid','delta')
    
    return graph
