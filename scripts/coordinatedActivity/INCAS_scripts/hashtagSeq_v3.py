import pandas as pd
import numpy as np
import math as mt
import networkx as nx
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

# Importing nltk
from nltk.corpus import stopwords
import nltk
import datetime

# importing regex
import re

# Data assumptions:
# minHashtags: minimum number of hashtags inside an hashtag sequence
# Index(['annotations', 'dataTags', 'embeddedUrls', 'extraAttributes',
#        'imageUrls', 'segments', 'author', 'contentText', 'id', 'language',
#        'mediaType', 'mediaTypeAttributes', 'name', 'timePublished', 'title',
#        'url', 'tweetid', 'retweet_id', 'engagementType', 'source_data',
#        'userid'],


#Downloading Stopwords
nltk.download('stopwords')

#Load English Stop Words
stopword = stopwords.words('english')

def preprocess_text(df):
    # Cleaning tweets in en language
    # Removing RT Word from Messages
    df['contentText']=df['contentText'].str.lstrip('RT')
    # Removing selected punctuation marks from Messages
    df['contentText']=df['contentText'].str.replace( ":",'')
    df['contentText']=df['contentText'].str.replace( ";",'')
    df['contentText']=df['contentText'].str.replace( ".",'')
    df['contentText']=df['contentText'].str.replace( ",",'')
    df['contentText']=df['contentText'].str.replace( "!",'')
    df['contentText']=df['contentText'].str.replace( "&",'')
    df['contentText']=df['contentText'].str.replace( "-",'')
    df['contentText']=df['contentText'].str.replace( "_",'')
    df['contentText']=df['contentText'].str.replace( "$",'')
    df['contentText']=df['contentText'].str.replace( "/",'')
    df['contentText']=df['contentText'].str.replace( "?",'')
    df['contentText']=df['contentText'].str.replace( "''",'')
    # Lowercase
    df['contentText']=df['contentText'].str.lower()

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


# retrieves tweet's timestamp from its ID
def get_tweet_timestamp(tid):
    try:
        offset = 1288834974657
        tstamp = (tid >> 22) + offset
        utcdttime = datetime.utcfromtimestamp(tstamp/1000)
        return utcdttime
    except:
        return None  


def hashSeq(cum,minHashtags = 3):
    cum = cum.loc[cum['engagementType'] != 'retweet']
    cum = preprocess_text(cum)
    cum['contentText'] = cum['contentText'].astype(str).apply(lambda x: msg_clean(x))


    cum['hashtag_seq'] = ['__'.join([tag.strip("#") for tag in tweet.split() if tag.startswith("#")]) for tweet in cum['contentText'].values.astype(str)]
    cum.drop('contentText', axis=1, inplace=True)
    cum = cum[['author', 'hashtag_seq']].loc[cum['hashtag_seq'].apply(lambda x: len(x.split('__'))) >= minHashtags]
    
    cum.drop_duplicates(inplace=True)
    
    temp = cum.groupby('hashtag_seq', as_index=False).count()
    cum = cum.loc[cum['hashtag_seq'].isin(temp.loc[temp['author']>1]['hashtag_seq'].to_list())]

    cum['value'] = 1
    
    hashs = dict(zip(list(cum.hashtag_seq.unique()), list(range(cum.hashtag_seq.unique().shape[0]))))
    cum['hashtag_seq'] = cum['hashtag_seq'].apply(lambda x: hashs[x]).astype(int)
    del hashs

    userid = dict(zip(list(cum.author.astype(str).unique()), list(range(cum.author.unique().shape[0]))))
    cum['author'] = cum['author'].astype(str).apply(lambda x: userid[x]).astype(int)
    
    person_c = pd.CategoricalDtype(sorted(cum.author.unique()), ordered=True)
    thing_c = pd.CategoricalDtype(sorted(cum.hashtag_seq.unique()), ordered=True)
    
    row = cum.author.astype(person_c).cat.codes
    col = cum.hashtag_seq.astype(thing_c).cat.codes
    sparse_matrix = csr_matrix((cum["value"], (row, col)), shape=(person_c.categories.size, thing_c.categories.size))
    del row, col, person_c, thing_c
    
    vectorizer = TfidfTransformer()
    tfidf_matrix = vectorizer.fit_transform(sparse_matrix)
    similarities = cosine_similarity(tfidf_matrix, dense_output=False)

    df_adj = pd.DataFrame(similarities.toarray())
    del similarities
    df_adj.index = userid.keys()
    df_adj.columns = userid.keys()
    G = nx.from_pandas_adjacency(df_adj)
    del df_adj
    
    G.remove_nodes_from(list(nx.isolates(G)))

    return G
