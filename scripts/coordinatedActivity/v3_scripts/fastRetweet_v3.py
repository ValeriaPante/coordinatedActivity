import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime
from datetime import timedelta

import warnings

# retrieves tweet's timestamp from its ID
def get_tweet_timestamp(tid):
#    try:
#        offset = 1288834974657
#        tstamp = (tid >> 22) + offset
#        utcdttime = datetime.utcfromtimestamp(tstamp/1000)
#        return utcdttime
#    except:
#        return None  

    offset = 1288834974657
    tstamp = (tid >> 22) + offset
    utcdttime = datetime.utcfromtimestamp(tstamp/1000)
    return utcdttime


# Data assumptions:
#   - 2 Pandas dataframes
#     - control: control dataset -> includes only columns ['user', 'retweeted_status', 'id']
#     - treated: information Operation dataset -> includes only columns ['tweetid', 'userid', 'retweet_tweetid', 'retweet_userid']
# timeInterval: time distance in seconds between retweet and original tweet under which a retweet is considered fast

def fastRetweet(control, treated, timeInterval = 10):
    control.dropna(inplace=True)
    treated.dropna(inplace=True)

    #control['retweet_id'] = control['retweeted_status'].apply(lambda x: int(dict(x)['id']))
    #control['retweet_userid'] = control['retweeted_status'].apply(lambda x: int(dict(dict(x)['user'])['id']))
    #control['userid'] = control['user'].apply(lambda x: int(dict(x)['id']))
    
    control['retweet_id'] = control['retweeted_status'].apply(lambda x: int(eval(x)['id']))
    control['retweet_userid'] = control['retweeted_status'].apply(lambda x: int(dict(eval(x)['user'])['id']))
    control['userid'] = control['user'].apply(lambda x: int(eval(x)['id']))
    
    control['tweet_timestamp'] = control['id'].apply(lambda x: get_tweet_timestamp(int(x)))
    control['retweet_timestamp'] = control['retweet_id'].apply(lambda x: get_tweet_timestamp(int(x)))
    control = control[['id', 'userid', 'retweet_id', 'tweet_timestamp', 'retweet_timestamp', 'retweet_userid']]
    control.columns = ['tweetid', 'userid', 'retweet_tweetid', 'tweet_timestamp', 'retweet_timestamp', 'retweet_userid']
    
    #print("tweet",len(control["tweet_timestamp"].values))
    
    treated['retweet_tweetid'] = treated['retweet_tweetid'].astype(int)
    treated['tweet_timestamp'] = treated['tweetid'].apply(lambda x: get_tweet_timestamp(int(x)))
    treated['retweet_timestamp'] = treated['retweet_tweetid'].apply(lambda x: get_tweet_timestamp(int(x)))

    treated['delta'] = (treated['tweet_timestamp'] - treated['retweet_timestamp']).dt.seconds
    control['delta'] = (control['tweet_timestamp'] - control['retweet_timestamp']).dt.seconds

        
    cumulative = pd.concat([treated[['userid', 'retweet_userid', 'delta']], control[['userid','retweet_userid', 'delta']]])
    cumulative['userid'] = cumulative['userid'].astype(int).astype(str)
    cumulative = cumulative.loc[cumulative['delta'] <= timeInterval]
    
    cumulative = cumulative.groupby(['userid', 'retweet_userid'],as_index=False).count()
    cumulative = cumulative.loc[cumulative['delta'] > 1]
    
    graph = nx.from_pandas_edgelist(cumulative, 'userid', 'retweet_userid','delta')
    graph.remove_edges_from(list(nx.selfloop_edges(graph)))
    
    warnings.warn(str(len((set(graph.nodes)))))

    return graph