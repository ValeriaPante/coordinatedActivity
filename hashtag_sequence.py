import numpy as np
import pandas as pd
import networkx as nx
from datetime import datetime
from datetime import timedelta

# Data assumptions:
#   - Pandas dataframe
#   - 'contentText' shows text of tweets
#   - 'twitterAuthorScreenname' is the label of user IDs
#   - 'engagementType' is the type of engagement (quote tweet, retweet, reply, or tweet)
#   - 'engagementParentId': is the tweet ID a tweet is replying, retweeting, or quote tweeting

def hashtag_coord(twitter_data, min_hashes = 5):
    # minimum of 5 hashtags; alternatives based on cosine similarity of tweets could work too
    twitter_data['hashtag_seq'] = twitter_data['contentText'].astype(str).apply(lambda text: '__'.join([tag.strip("#") for tag in text.split() if tag.startswith("#")]))
    unique_hash_seq = twitter_data['hashtag_seq'].drop_duplicates()
    hashes = twitter_data.groupby('hashtag_seq')
    duplicate_hash_users = {}

    for jj,tweet in enumerate(unique_hash_seq):
        
        if len(tweet.split('__')) < min_hashes: continue
        try:
            all_tweets = hashes.get_group(tweet)
        except:
            continue
        all_tweets_tweet = all_tweets.loc[all_tweets['engagementType']=='tweet',]
        num_users = len(all_tweets_tweet['twitterAuthorScreenname'].drop_duplicates())
        # if multiple tweets and multiple users 
        if num_users < len(all_tweets_tweet):
            links = all_tweets[['tweetId','engagementParentId']].drop_duplicates()
            users = all_tweets['twitterAuthorScreenname'].drop_duplicates().tolist()
            duplicate_hash_users[tweet] = users
    all_dup_hash_users = []
    coord_hash_users = []
    for key in duplicate_hash_users.keys():
        if len(duplicate_hash_users[key]) > 1:
            coord_hash_users.append(duplicate_hash_users[key])
        all_dup_hash_users+=duplicate_hash_users[key]
    # network
    edges = []
    for nodes in coord_hash_users:
        unique_edges = list(set([tuple(sorted([n1,n2])) for n1 in 
    nodes for n2 in nodes if n1 != n2]))
        edges+=(unique_edges)
    # nodes = all user IDs of coorinated users
    # edges = if these pairs of users are coordinated
    G = nx.Graph()
    G.add_edges_from(edges)
    return G
