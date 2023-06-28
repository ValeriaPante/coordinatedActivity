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
# timeInterval: time distance between retweet and original tweet under which a retweet is considered fast
    
def fast_retweet(data, timeInterval):

    data['tweetid'] = data['tweetid'].astype(int)
    data['retweet_tweetid'] = data['retweet_tweetid'].astype(int)
    data['tweetTime'] = data['tweetid'].apply(lambda x: get_tweet_timestamp(x))
    data['original_ts'] = data['retweet_tweetid'].apply(lambda x: get_tweet_timestamp(x))
    
    data['timeDifference'] = data['tweetTime'] - data['original_ts']
    data['predicted'] = data['timeDifference'].apply(lambda x: 1 if x <= timedelta(seconds=timeInterval) else 0)
    
    return data[['userid', 'predicted']].groupby('userid', as_index=False).max()
