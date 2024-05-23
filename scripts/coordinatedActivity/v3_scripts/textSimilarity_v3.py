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

# MAIN FUNCTION
# Data assumptions:
#   - datasetsPaths: list containing the absolute paths referring to the datasets to analyze (no distiction between control and information operations ones)
#   - outputDir: directory where to save temporary files
# To solve computational issues, the function will create multiple output files of users sharing similar texts that will need to then be merged into a network using the getSimilarityNetwork function (see below)

def textSim(datasetsPaths, outputDir):
    
    
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
    
    for file in datasetsPaths:
        if 'control' in file:
            if file[-2:] == 'gz':
                try:
                    with gzip.open(file) as f:
                        control = pd.concat([control, get_negative_data(pd.read_json(f, lines =True)[['id', 'full_text', 'lang', 'user', 'created_at']])])
                except:
                    with gzip.open(file) as f:
                        control = get_negative_data(pd.read_json(f, lines =True)[['id', 'full_text', 'lang', 'user', 'created_at']])
            else:
                try:
                    with open(file) as f:
                        control = pd.concat([control, get_negative_data(pd.read_json(f, lines =True)[['id', 'full_text', 'lang', 'user', 'created_at']])])
                except:
                    with open(file) as f:
                        control = get_negative_data(pd.read_json(f, lines =True)[['id', 'full_text', 'lang', 'user', 'created_at']])
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
    
            
    pos_en_df_all = preprocess_text(treated)
    del treated
    neg_en_df_all = preprocess_text(control)
    del control
    
    pos_en_df_all['tweet_text']  = pos_en_df_all['tweet_text'].replace(',', '')
    neg_en_df_all['tweet_text']  = neg_en_df_all['tweet_text'].replace(',', '')
    
    pos_en_df_all['clean_tweet'] = pos_en_df_all['tweet_text'].astype(str).apply(lambda x: msg_clean(x))
    neg_en_df_all['clean_tweet'] = neg_en_df_all['tweet_text'].astype(str).apply(lambda x: msg_clean(x))
    
    pos_en_df_all = pos_en_df_all[pos_en_df_all['clean_tweet'].apply(lambda x: len(x.split(' ')) > 4)]
    neg_en_df_all = neg_en_df_all[neg_en_df_all['clean_tweet'].apply(lambda x: len(x.split(' ')) > 4)]

    pos_en_df_all['tweet_time'] = pos_en_df_all['tweetid'].apply(lambda x: get_tweet_timestamp(x))
    neg_en_df_all['tweet_time'] = neg_en_df_all['tweetid'].apply(lambda x: get_tweet_timestamp(x))
    
    date = pos_en_df_all['tweet_time'].min().date()
    finalDate = pos_en_df_all['tweet_time'].max().date()
    
    i = 1

    combined_tweets_df = None
    
    while date <= finalDate:
        
        pos_en_df = pos_en_df_all.loc[(pos_en_df_all['tweet_time'].dt.date >= date)&(pos_en_df_all['tweet_time'].dt.date < date+timedelta(days=1))]
        neg_en_df = neg_en_df_all.loc[(neg_en_df_all['tweet_time'].dt.date >= date)&(neg_en_df_all['tweet_time'].dt.date < date+timedelta(days=1))]
    
        actual_pos_user = pos_en_df.userid.unique()
        actual_neg_user = neg_en_df.userid.unique()
    
        combined_tweets_df = pd.concat([pos_en_df, neg_en_df], axis=0)
        combined_tweets_df.reset_index(inplace=True)
        combined_tweets_df = combined_tweets_df.loc[:, ~combined_tweets_df.columns.str.contains('index')]
    
        del pos_en_df
        del neg_en_df
    
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
    
        sim_score_df = create_sim_score_df(lims,D,I,search_query1)
        print('Generated Similarity Score DataFrame')
    
        del combined_tweets_df
    
        for threshold in np.arange(0.7,1.01,0.05):
    
            print("Threshold: ", threshold)
    
            sim_score_temp_df = sim_score_df[sim_score_df.sim_score >= threshold]
    
            text_sim_network = sim_score_temp_df[['source_user','target_user']]
            text_sim_network = text_sim_network.drop_duplicates(subset=['source_user','target_user'], keep='first')
    
            outputfile = outputDir + '/threshold_' + str(threshold) + '_'+str(i)+'.csv'
            text_sim_network.to_csv(outputfile)

        
        date = date+timedelta(days=1)
        i += 1

# to run after the textSim function
# inputDir: path of the directory containing the similarity files; it corresponds to the outputDir used in the textSim function
def getSimilarityNetwork(inputDir):
    
    # Warnings
    warnings.warn("Similarity Network")
    
    files = [f for f in listdir(inputDir)]
    files.sort()

    d = {'threshold_1.00':[],
        'threshold_0.90':[],
        'threshold_0.95':[],
        'threshold_0.85':[],
        'threshold_0.8':[],
        'threshold_0.75':[],
        'threshold_0.7':[]}
    
    for f in files:
        if f[:9]=='threshold':
            d['_'.join(f[:-4].split('_')[:2])[:14]].append(f)

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
    combined.to_csv("/scratch1/ashwinba/cache/text_sim_temp.csv")

    G = nx.from_pandas_edgelist(combined, source='source_user', target='target_user', edge_attr=['weight'])
    
    nx.write_gml(G,"/scratch1/ashwinba/cache/text_similarity.gml.gz")
    warnings.warn("written gml file")
    
    MAPPER_DIR = "/scratch1/ashwinba/consolidated"
    CONTROL_MAPPER_DIR = "control_consolidated_raw.csv.gz"
    TREATED_MAPPER_DIR = "treated_consolidated_raw.csv.gz"
    
    control = pd.read_csv(os.path.join(MAPPER_DIR,CONTROL_MAPPER_DIR),compression='gzip')
    treated = pd.read_csv(os.path.join(MAPPER_DIR,TREATED_MAPPER_DIR),compression='gzip')
    
    # Dropping NaN Values
    control.dropna(subset=['user','country'],inplace=True)
    #treated.dropna(subset=['userid','country'],inplace=True)
    
    control['userid'] = control['user'].apply(lambda x: str(int(eval(x)['id'].strip())))
    
    mapper1 = control[['country','userid']]
    mapper2 = treated[['country','userid']]
    
    # Deleting dfs
    del control
    del treated
    
    #mapper1['userid'] = mapper1['userid'].astype(str)
    #mapper2['userid'] = mapper2['userid'].astype(str)
    
    # Calling Function
    generate_country_wise(mapper1,mapper2,G)
    
            
    return G
    

def generate_country_wise(mapper1,mapper2,graph):
    # Generate Vertices
    graph = nx.relabel_nodes(graph, lambda x: str(x))
    graph.remove_edges_from(list(nx.selfloop_edges(graph)))
    vertices = list(set(graph.nodes))
    warnings.warn("Total Vertices :"+str(len(vertices)))
    
    mapper1['class_label'] = "control"
    mapper2['class_label'] = "treated"
    
    #countries = list(set(list(set(list(mapper1['country'].values) + list(mapper2['country'].values)))))

    merged = pd.concat([mapper1,mapper2])
    
    #Delete NaN Values
    merged.dropna(subset=['userid'],inplace=True)
    
    merged['userid'] = merged['userid'].apply(lambda x: str(x))
    warnings.warn("userid dtype changed")
    
    # Deleting Consolidated Files
    del mapper1
    del mapper2
    
    attributes = {}
    counter = 0
    warnings.warn("Starting to work")
    for v in vertices:
        counter+=1
        if(counter%10 == 0):
            warnings.warn(str(counter) + " work done")
        temp_dict = {}
        
        if((merged['userid'] == v).any()):
            temp_dict['class_label'] = merged[merged.userid == v].iloc[0]['class_label']

        else:
            warnings.warn(str(v)+" not found")
            temp_dict["class_label"] = "treated"
        
        temp_dict["userid"] = str(v)
        country_mapped = merged.loc[merged['userid'] == v,'country']
        if(len(country_mapped) == 0 or country_mapped.values[0] in ["egypt","uae"]):
            temp_dict["country"] = "Egypt&UAE"
        else:
            temp_dict["country"] = country_mapped.values[0]

        attributes[v] = temp_dict
            
  
    # Save
    nx.set_node_attributes(graph,attributes)
    nx.draw(graph, with_labels=True, font_weight='normal')
    nx.write_gexf(graph,"/scratch1/ashwinba/cache/plt_countrywise_textsim.gexf")
    #plt.savefig("/scratch1/ashwinba/cache/plt_countrywise_textsim_img.png")

    print("Executed")
        