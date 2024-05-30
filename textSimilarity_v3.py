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
import shutil

import warnings

# MAIN FUNCTION at line 199

def get_tweet_timestamp(tid):
    try:
        # offset = 1288834974657
        # tstamp = (tid >> 22) + offset
        utcdttime = datetime.utcfromtimestamp(tid/1000)
        return utcdttime
    except:
        return None  

#Downloading Stopwords
nltk.download('stopwords')

#Load English Stop Words
stopword = stopwords.words('english')

combined_tweets_df = None


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
    
    
    warnings.warn(str(df.columns))
    
    df = df[df['tweet_type'] != 'retweet']

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
    global combined_tweets_df
    
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


# MAIN FUNCTION
# Data assumptions:
#   - datasetsPaths: list containing the absolute paths referring to the datasets to analyze (no distiction between control and information operations ones)
#   - outputDir: directory where to save temporary files
# To solve computational issues, the function will create multiple output files of users sharing similar texts that will need to then be merged into a network using the getSimilarityNetwork function (see below)

# Index(['annotations', 'dataTags', 'embeddedUrls', 'extraAttributes',
#        'imageUrls', 'segments', 'author', 'contentText', 'geolocation', 'id',
#        'language', 'mediaType', 'mediaTypeAttributes', 'mentionedUsers',
#        'name', 'timePublished', 'title', 'url', 'translatedContentText',
#        'translatedTitle','engagementType','tweetid'],
#       dtype='object'

# Mandatory Fields
# 1. tweet_text
# ['userid' --> numerical encoding of author]
def textSim(cum,outputDir):
    global combined_tweets_df

    # Creating output dir if not exists
    if os.path.exists(outputDir):
        shutil.rmtree(outputDir)
        
    os.mkdir(outputDir)

    warnings.warn(str(cum.columns))
    cum = cum[cum['engagementType'] != 'retweet']

    # Changing colummns
    cum.rename(columns={'engagementType':'tweet_type','contentText':'tweet_text'},inplace=True)

    # Adding Timestamp
    #cum['tweet_time'] = cum['tweetid'].apply(lambda x: get_tweet_timestamp(x))
    cum['tweet_time'] = cum['timePublished'].apply(lambda x:get_tweet_timestamp(x))
    warnings.warn("calculated tweet_time")
    
    # Preprocess tweet texts
    cum_all = preprocess_text(cum)
    cum_all['tweet_text'] = cum['tweet_text'].replace(',','')
    cum_all['clean_tweet'] = cum['tweet_text'].astype(str).apply(lambda x:msg_clean(x))

    # Cleaning text
    cum_all = cum_all[cum_all['clean_tweet'].apply(lambda x: len(x.split(' ')) > 4)]
    
    #print(cum_all.shape)

    date = cum_all['tweet_time'].min().date()
    finalDate = cum_all['tweet_time'].max().date()
    
    i = 1
    while date <= finalDate:
        cum_all1 = cum_all.loc[(cum_all['tweet_time'].dt.date >=date)&(cum_all['tweet_time'].dt.date < date + timedelta(days=1))]
        actual_user = cum_all.userid.unique()

        combined_tweets_df = cum_all1.copy()
        combined_tweets_df.reset_index(inplace=True)
        combined_tweets_df = combined_tweets_df.loc[:, ~combined_tweets_df.columns.str.contains('index')]
    
        del cum_all1
    
        combined_tweets_df.reset_index(inplace=True)
        combined_tweets_df = combined_tweets_df.rename(columns = {'index':'my_idx'})
    
        sentences = combined_tweets_df.clean_tweet.tolist()
    
        encoder = SentenceTransformer('stsb-xlm-r-multilingual')
        plot_embeddings = encoder.encode(sentences)    

        try:
            dim = plot_embeddings.shape[1]  # vector dimension
        except:
            date = date+timedelta(days=1)
            continue
    
        db_vectors1 = plot_embeddings.copy().astype(np.float32)
        a = [i for i in range(plot_embeddings.shape[0])]
        db_ids1 = np.array(a, dtype=np.int64)
    
        faiss.normalize_L2(db_vectors1)
    
        index1 = faiss.IndexFlatIP(dim)
        index1 = faiss.IndexIDMap(index1)  # mapping df index as id
        index1.add_with_ids(db_vectors1, db_ids1)
    
        search_query1 = plot_embeddings.copy().astype(np.float32)
    
        faiss.normalize_L2(search_query1)

        result_plot_thres = []
        result_plot_score = []
        result_plot_metrics = []
    
        init_threshold = 0.7
    
        lims, D, I = index1.range_search(x=search_query1, thresh=init_threshold)
        print('Retrieved results of index search')
        warnings.warn('Retrieved results of index search')
    
        sim_score_df = create_sim_score_df(lims,D,I,search_query1)
        print('Generated Similarity Score DataFrame')
        warnings.warn('Generated Similarity Score DataFrame')
    
        del combined_tweets_df
        
        # for threshold in np.arange(0.7,1.01,0.05):
        for threshold in [0.95]:    
            print("Threshold: ", threshold)
    
            sim_score_temp_df = sim_score_df[sim_score_df.sim_score >= threshold]
    
            text_sim_network = sim_score_temp_df[['source_user','target_user']]
            text_sim_network = text_sim_network.drop_duplicates(subset=['source_user','target_user'], keep='first')
    
            outputfile = outputDir + '/threshold_' + str(threshold) + '_'+str(i)+'.csv'
            text_sim_network.to_csv(outputfile)

        date = date+timedelta(days=1)
        i += 1
        warnings.warn(str(i))


# to run after the textSim function
# inputDir: path of the directory containing the similarity files; it corresponds to the outputDir used in the textSim function
def getSimilarityNetwork(inputDir):

    global combined_tweets_df
    
    # Warnings
    warnings.warn("Similarity Network")
    
    files = [f for f in listdir(inputDir)]
    files.sort()
    
    
    # Aggregate thresholds
    d = {'threshold_1.00':[],
        'threshold_0.90':[],
        'threshold_0.95':[],
        'threshold_0.85':[],
        'threshold_0.8':[],
        'threshold_0.75':[],
        'threshold_0.7':[]}
        
    # Particular Threshold
    d = {'threshold_0.95':[]}
    
    for f in files:
        try:
            if f[:9]=='threshold':
                d['_'.join(f[:-4].split('_')[:2])[:14]].append(f)
        except Exception as e:
            pass

    i = 0

    for fil in d.keys():
        thr = float(fil.split('_')[-1][:4])
        
        l = d[fil]
        if i == 0:
            combined = pd.read_csv(os.path.join(inputDir,l[0]))
            # Dropping NaN Records
            combined.dropna(inplace=True)
            combined['weight'] = thr
            combined = combined[['weight','source_user','target_user']]
            i += 1
            for o in l[1:]:
                temp = pd.read_csv(os.path.join(inputDir,o))
                temp['weight'] = thr
                combined = pd.concat([combined, temp],ignore_index=True)
        else:
            for o in l:
                temp = pd.read_csv(os.path.join(inputDir,o))
                temp['weight'] = thr
                combined = pd.concat([combined, temp],ignore_index=True)

    combined['source_user'] = combined['source_user'].apply(lambda x: str(x).strip())
    combined['target_user'] = combined['target_user'].apply(lambda x: str(x).strip())
    
    
    combined.sort_values(by='weight', ascending=False, inplace=True)
    combined.drop_duplicates(subset=['source_user', 'target_user'], inplace=True)
    
    warnings.warn("written csv file")
    combined.to_csv("/scratch1/ashwinba/cache/INCAS/text_sim_temp.csv")

    G = nx.from_pandas_edgelist(combined, source='source_user', target='target_user', edge_attr=['weight'])
    
    warnings.warn("written gml file")

    return G