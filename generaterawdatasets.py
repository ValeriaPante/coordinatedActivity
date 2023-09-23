from pathlib import Path
import pandas as pd
import numpy as np
import ast
from tqdm import tqdm
import os
from os import listdir
from os.path import isfile, join, isdir
from itertools import compress
import itertools
from tabulate import tabulate
import re
import warnings
warnings.filterwarnings("ignore")
from scipy import spatial
import json
import gzip


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
    #neg_df = pd.concat([neg_df, pd.DataFrame(list(neg_df['user']))['id']], axis=1)
    neg_df.rename(columns = {'full_text':'tweet_text'}, inplace = True)
    return neg_df

def get_positive_data(pos_df):
    #pos_df = process_data(pos_df)
    pos_df['tweet_time'] = pos_df['tweetid'].apply(lambda x: get_tweet_timestamp(x))
    
    return pos_df


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

def GenerateDatasets(datasetsPaths):
    for file in datasetsPaths:
        if 'control' in file:
            if file[-2:] == 'gz':
                try:
                    with gzip.open(file) as f:
                        control = pd.concat([control, get_negative_data(pd.read_json(f, lines =True))])
                except:
                    with gzip.open(file) as f:
                        control = get_negative_data(pd.read_json(f, lines =True))
            else:
                try:
                    with open(file) as f:
                        control = pd.concat([control, get_negative_data(pd.read_json(f, lines =True))])
                except:
                    with open(file) as f:
                        control = get_negative_data(pd.read_json(f, lines =True))
        else:
            if file[-2:] == 'gz':
                try:
                    with gzip.open(file) as f:
                        treated = pd.concat([treated, get_positive_data(pd.read_csv(f))])
                except:
                    with gzip.open(file) as f:
                        treated = get_positive_data(pd.read_csv(f))
            else:
                try:
                    with open(file) as f:
                        treated = pd.concat([treated, get_positive_data(pd.read_csv(f))])
                except:
                    with open(file) as f:
                        treated = get_positive_data(pd.read_csv(f))
    
            
    pos_en_df_all = treated
    del treated
    neg_en_df_all = control
    del control
    
    pos_en_df_all['tweet_text']  = pos_en_df_all['tweet_text'].replace(',', '')
    neg_en_df_all['tweet_text']  = neg_en_df_all['tweet_text'].replace(',', '')

    pos_en_df_all.to_csv("/scratch1/ashwinba/consolidated/treated_consolidated_raw.csv.gz", index=False, compression='gzip')
    neg_en_df_all.to_csv("/scratch1/ashwinba/consolidated/control_consolidated_raw.csv.gz", index=False, compression='gzip')

root_dir = "/project/ll_774_951/InfoOpsNationwiseDriverControl"
countries_dir = os.listdir("/project/ll_774_951/InfoOpsNationwiseDriverControl")

dataset_dirs = []
print("I am working")
for country in countries_dir:
    print(country)
    files_dir = os.listdir(os.path.join(root_dir,country))

    ## Country File Names Check
    control_check = list(filter(lambda x:"control" in x,files_dir))
    treated_check = list(filter(lambda x:"tweets_csv_unhashed" in x,files_dir))
    
    if(len(control_check) >= 1 and len(treated_check) >= 1):
        dataset_dirs.append(os.path.join(root_dir,country,treated_check[0]))
        dataset_dirs.append(os.path.join(root_dir,country,control_check[0]))

GenerateDatasets(dataset_dirs)