import torch
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import ast
from tqdm import tqdm
import os
from os import listdir
from os.path import isfile, join, isdir
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import compress
import itertools
from tabulate import tabulate
import plotly.graph_objects as go
from nltk.corpus import stopwords
import nltk
import re
import warnings
#warnings.filterwarnings("ignore")
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy import spatial
import json
import gzip
import glob
from datetime import datetime
from datetime import timedelta
import networkx as nx 

import warnings

# MAIN FUNCTION at line 199

def get_tweet_timestamp(tid):
    try:
        offset = 1288834974657
        tstamp = (tid >> 22) + offset
        utcdttime = datetime.utcfromtimestamp(tstamp/1000)
        return utcdttime
    except:
        return None  

def get_negative_data(neg_df):
    neg_df = pd.concat([neg_df, pd.DataFrame(list(neg_df['user']))['id']], axis=1)
    neg_df.drop('user', axis=1, inplace=True)
    neg_df.columns = ['tweetid','tweet_text','tweet_language','tweet_time','userid']

    return neg_df

def process_data(tweet_df):
    tweet_df['quoted_tweet_tweetid'] = tweet_df['quoted_tweet_tweetid'].astype('Int64')
    tweet_df['retweet_tweetid'] = tweet_df['retweet_tweetid'].astype('Int64')
    
    # Tweet type classification
    tweet_type = []
    for i in range(tweet_df.shape[0]):
        if pd.notnull(tweet_df['quoted_tweet_tweetid'].iloc[i]):
            if pd.notnull(tweet_df['retweet_tweetid'].iloc[i]):
                if pd.notnull(tweet_df['in_reply_to_tweetid'].iloc[i]):
                    continue
                else:
                    tweet_type.append('retweet')
            else:
                if pd.notnull(tweet_df['in_reply_to_tweetid'].iloc[i]):
                    tweet_type.append('reply')
                else:
                    tweet_type.append('quoted')
        else:
            if pd.notnull(tweet_df['retweet_tweetid'].iloc[i]):
                if pd.notnull(tweet_df['in_reply_to_tweetid'].iloc[i]):
                    continue
                else:
                    tweet_type.append('retweet')
            else:
                if pd.notnull(tweet_df['in_reply_to_tweetid'].iloc[i]):
                    tweet_type.append('reply')
                else:
                    tweet_type.append('original')
    tweet_df['tweet_type'] = tweet_type
    tweet_df = tweet_df[tweet_df.tweet_type != 'retweet']
    
    return tweet_df

def get_positive_data(pos_df):
    pos_df = process_data(pos_df)
    pos_df = pos_df[['tweetid','userid','tweet_time','tweet_language','tweet_text']]
    pos_df['tweet_time'] = pos_df['tweetid'].apply(lambda x: get_tweet_timestamp(x))
    
    return pos_df

#Downloading Stopwords
nltk.download('stopwords')

#Load English Stop Words
stopword = stopwords.words('english')

def preprocess_text(df):
    # Cleaning tweets in en language
    # Removing RT Word from Messages
    df['tweet_text']=df['tweet_text'].str.lstrip('RT')
    # Removing selected punctuation marks from Messages
    df['tweet_text']=df['tweet_text'].str.replace( ":",'')
    df['tweet_text']=df['tweet_text'].str.replace( ";",'')
    df['tweet_text']=df['tweet_text'].str.replace( ".",'')
    df['tweet_text']=df['tweet_text'].str.replace( ",",'')
    df['tweet_text']=df['tweet_text'].str.replace( "!",'')
    df['tweet_text']=df['tweet_text'].str.replace( "&",'')
    df['tweet_text']=df['tweet_text'].str.replace( "-",'')
    df['tweet_text']=df['tweet_text'].str.replace( "_",'')
    df['tweet_text']=df['tweet_text'].str.replace( "$",'')
    df['tweet_text']=df['tweet_text'].str.replace( "/",'')
    df['tweet_text']=df['tweet_text'].str.replace( "?",'')
    df['tweet_text']=df['tweet_text'].str.replace( "''",'')
    # Lowercase
    df['tweet_text']=df['tweet_text'].str.lower()

    return df

def remove_emoji(string):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

#Message Clean Function
def msg_clean(msg):
    #Remove URL
    msg = re.sub(r'https?://\S+|www\.\S+', " ", msg)

    #Remove Mentions
    msg = re.sub(r'@\w+',' ',msg)

    #Remove Digits
    msg = re.sub(r'\d+', ' ', msg)

    #Remove HTML tags
    msg = re.sub('r<.*?>',' ', msg)
    
    #Remove HTML tags
    msg = re.sub('r<.*?>',' ', msg)
    
    #Remove Emoji from text
    msg = remove_emoji(msg)

    # Remove Stop Words 
    msg = msg.split()
    
    msg = " ".join([word for word in msg if word not in stopword])

    return msg

def create_sim_score_df(lims,D,I,search_query1):
    source_idx = []
    target_idx = []
    sim_score = []

    for i in range(len(search_query1)):
        idx = I[lims[i]:lims[i+1]]
        sim = D[lims[i]:lims[i+1]]
        for j in range(len(idx)):
            source_idx.append(i)
            target_idx.append(idx[j])
            sim_score.append(sim[j])

    sim_score_df = pd.DataFrame(list(zip(source_idx, target_idx, sim_score)), columns=['source_idx', 'target_idx', 'sim_score'])
    del source_idx
    del target_idx
    del sim_score
    sim_score_df = sim_score_df.query("source_idx != target_idx")
    sim_score_df['combined_idx'] = sim_score_df[['source_idx', 'target_idx']].apply(tuple, axis=1)
    sim_score_df['combined_idx'] = sim_score_df['combined_idx'].apply(sorted)
    sim_score_df['combined_idx'] = sim_score_df['combined_idx'].transform(lambda k: tuple(k))
    sim_score_df = sim_score_df.drop_duplicates(subset=['combined_idx'], keep='first')
    sim_score_df.reset_index(inplace=True)
    sim_score_df = sim_score_df.loc[:, ~sim_score_df.columns.str.contains('index')]
    sim_score_df.drop(['combined_idx'], inplace = True, axis=1)

    df_join = pd.merge(pd.merge(sim_score_df,combined_tweets_df, left_on='source_idx', right_on='my_idx', how='inner'),combined_tweets_df,left_on='target_idx',right_on='my_idx',how='inner')

    result = df_join[['userid_x','userid_y','clean_tweet_x','clean_tweet_y','sim_score']]
    result = result.rename(columns = {'userid_x':'source_user',
                                     'userid_y':'target_user',
                                     'clean_tweet_x':'source_text',
                                     'clean_tweet_y':'target_text'})
    return result
