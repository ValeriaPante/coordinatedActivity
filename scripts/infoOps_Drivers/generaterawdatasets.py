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

def get_negative_data(neg_df,country):
    #neg_df = pd.concat([neg_df, pd.DataFrame(list(neg_df['user']))['id']], axis=1)
    neg_df.rename(columns = {'full_text':'tweet_text'}, inplace = True)
    neg_df['country'] = country
    return neg_df

def get_positive_data(pos_df,country):
    #pos_df = process_data(pos_df)
    pos_df['tweet_time'] = pos_df['tweetid'].apply(lambda x: get_tweet_timestamp(x))
    pos_df['country'] = country
    
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
    control = pd.DataFrame()
    treated=  pd.DataFrame()
    for record in datasetsPaths:
        country,file = record
        if 'control' in file:
            if file[-2:] == 'gz':
                try:
                    control = pd.concat([control, get_negative_data(pd.read_json(file, lines =True,compression='gzip'),country)])
                except:
                    control = get_negative_data(pd.read_json(file, lines =True,compression='gzip'),country)
            else:
                try:
                    control = pd.concat([control, get_negative_data(pd.read_json(f, lines =True),country)])
                except:
                    control = get_negative_data(pd.read_json(file, lines =True),country)
        else:
            if file[-2:] == 'gz':
                try:
                    treated = pd.concat([treated, get_positive_data(pd.read_csv(file,compression='gzip'),country)])
                except:
                    treated = get_positive_data(pd.read_csv(file,compression='gzip'),country)
            else:
                try:
                    treated = pd.concat([treated, get_positive_data(pd.read_csv(file),country)])
                except:
                    treated = get_positive_data(pd.read_csv(file),country)
        
        warnings.warn("Control Shape :"+str(control.shape))
        warnings.warn("Treated Shape :"+str(treated.shape))
        
    #pos_en_df_all = treated
    #del treated
    #neg_en_df_all = control
    #del control
    
    #treated['tweet_text']  = trea['tweet_text'].replace(',', '')
    #control['tweet_text']  = neg_en_df_all['tweet_text'].replace(',', '')
    
    print("FINAL SHAPE")
    warnings.warn("Control Shape :"+str(control.shape))
    warnings.warn("Treated Shape :"+str(treated.shape))
    
    print("COUNTRIES LIST")
    print("Control",len(control['country'].unique()))
    print("Treated",len(treated['country'].unique()))

    treated.to_csv("/project/muric_789/ashwin/consolidated/treated_consolidated_raw.csv.gz", index=False, compression='gzip')
    control.to_csv("/project/muric_789/ashwin/consolidated/control_consolidated_raw.csv.gz", index=False, compression='gzip')

root_dir = "/project/ll_774_951/InfoOpsNationwiseDriverControl"
countries_dir = os.listdir("/project/ll_774_951/InfoOpsNationwiseDriverControl")
countries_to_be_removed = ['venezuela_set2','egypt']
for c in countries_to_be_removed:countries_dir.remove(c)
    

dataset_dirs = []
for country in countries_dir:
    files_dir = os.listdir(os.path.join(root_dir,country))

    ## Country File Names Check
    control_check = list(filter(lambda x:"control" in x,files_dir))
    treated_check = list(filter(lambda x:"tweets_csv_unhashed" in x,files_dir))
    
    if(len(control_check) >= 1 and len(treated_check) >= 1):
        dataset_dirs.append([country,os.path.join(root_dir,country,treated_check[0])])
        dataset_dirs.append([country,os.path.join(root_dir,country,control_check[0])])

GenerateDatasets(dataset_dirs)