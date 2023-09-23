from datetime import datetime
import gzip
import pandas as pd
import os
from nltk.corpus import stopwords
import nltk
import re
import warnings
warnings.filterwarnings("ignore")

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
    neg_df = pd.concat([neg_df, pd.DataFrame(
        list(neg_df['user']))['id']], axis=1)
    neg_df.drop('user', axis=1, inplace=True)
    neg_df.columns = ['tweetid', 'tweet_text',
                      'tweet_language', 'tweet_time', 'userid']

    return neg_df


def get_positive_data(pos_df):
    pos_df = process_data(pos_df)
    pos_df = pos_df[['tweetid', 'userid',
                     'tweet_time', 'tweet_language', 'tweet_text']]
    pos_df['tweet_time'] = pos_df['tweetid'].apply(
        lambda x: get_tweet_timestamp(x))

    return pos_df


# Downloading Stopwords
nltk.download('stopwords')

# Load English Stop Words
stopword = stopwords.words('english')


def preprocess_text(df):
    # Cleaning tweets in en language
    # Removing RT Word from Messages
    df['tweet_text'] = df['tweet_text'].str.lstrip('RT')
    # Removing selected punctuation marks from Messages
    df['tweet_text'] = df['tweet_text'].str.replace(":", '')
    df['tweet_text'] = df['tweet_text'].str.replace(";", '')
    df['tweet_text'] = df['tweet_text'].str.replace(".", '')
    df['tweet_text'] = df['tweet_text'].str.replace(",", '')
    df['tweet_text'] = df['tweet_text'].str.replace("!", '')
    df['tweet_text'] = df['tweet_text'].str.replace("&", '')
    df['tweet_text'] = df['tweet_text'].str.replace("-", '')
    df['tweet_text'] = df['tweet_text'].str.replace("_", '')
    df['tweet_text'] = df['tweet_text'].str.replace("$", '')
    df['tweet_text'] = df['tweet_text'].str.replace("/", '')
    df['tweet_text'] = df['tweet_text'].str.replace("?", '')
    df['tweet_text'] = df['tweet_text'].str.replace("''", '')
    # Lowercase
    df['tweet_text'] = df['tweet_text'].str.lower()

    return df


def process_data(tweet_df):
    tweet_df['quoted_tweet_tweetid'] = tweet_df['quoted_tweet_tweetid'].astype(
        'Int64')
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

# Message Clean Function


def msg_clean(msg):
    # Remove URL
    msg = re.sub(r'https?://\S+|www\.\S+', " ", msg)

    # Remove Mentions
    msg = re.sub(r'@\w+', ' ', msg)

    # Remove Digits
    msg = re.sub(r'\d+', ' ', msg)

    # Remove HTML tags
    msg = re.sub('r<.*?>', ' ', msg)

    # Remove HTML tags
    msg = re.sub('r<.*?>', ' ', msg)

    # Remove Emoji from text
    msg = remove_emoji(msg)

    # Remove Stop Words
    msg = msg.split()

    msg = " ".join([word for word in msg if word not in stopword])

    return msg


def GenerateDatasets(datasetsPaths):
    for file in datasetsPaths:
        if 'control' in file:
            if file[-2:] == 'gz':
                try:
                    with gzip.open(file) as f:
                        control = pd.concat([control, get_negative_data(pd.read_json(f, lines=True)[
                                            ['id', 'full_text', 'lang', 'user', 'created_at']])])
                except:
                    with gzip.open(file) as f:
                        control = get_negative_data(pd.read_json(f, lines=True)[
                                                    ['id', 'full_text', 'lang', 'user', 'created_at']])
            else:
                try:
                    with open(file) as f:
                        control = pd.concat([control, get_negative_data(pd.read_json(f, lines=True)[
                                            ['id', 'full_text', 'lang', 'user', 'created_at']])])
                except:
                    with open(file) as f:
                        control = get_negative_data(pd.read_json(f, lines=True)[
                                                    ['id', 'full_text', 'lang', 'user', 'created_at']])
        else:
            if file[-2:] == 'gz':
                try:
                    with gzip.open(file) as f:
                        treated = pd.concat(
                            [treated, get_positive_data(pd.read_csv(f))])
                except:
                    with gzip.open(file) as f:
                        treated = get_positive_data(pd.read_csv(f))
            else:
                try:
                    with open(file) as f:
                        treated = pd.concat(
                            [treated, get_positive_data(pd.read_csv(f))])
                except:
                    with open(file) as f:
                        treated = get_positive_data(pd.read_csv(f))

    pos_en_df_all = preprocess_text(treated)
    del treated
    neg_en_df_all = preprocess_text(control)
    del control

    pos_en_df_all['tweet_text'] = pos_en_df_all['tweet_text'].replace(',', '')
    neg_en_df_all['tweet_text'] = neg_en_df_all['tweet_text'].replace(',', '')

    pos_en_df_all['clean_tweet'] = pos_en_df_all['tweet_text'].astype(
        str).apply(lambda x: msg_clean(x))
    neg_en_df_all['clean_tweet'] = neg_en_df_all['tweet_text'].astype(
        str).apply(lambda x: msg_clean(x))

    pos_en_df_all = pos_en_df_all[pos_en_df_all['clean_tweet'].apply(
        lambda x: len(x.split(' ')) > 4)]
    neg_en_df_all = neg_en_df_all[neg_en_df_all['clean_tweet'].apply(
        lambda x: len(x.split(' ')) > 4)]

    pos_en_df_all['tweet_time'] = pos_en_df_all['tweetid'].apply(
        lambda x: get_tweet_timestamp(x))
    neg_en_df_all['tweet_time'] = neg_en_df_all['tweetid'].apply(
        lambda x: get_tweet_timestamp(x))

    pos_en_df_all.to_csv(
        "/scratch1/ashwinba/consolidated/treated_consolidated.csv")
    neg_en_df_all.to_csv(
        "/scratch1/ashwinba/consolidated/control_consolidated.csv")


root_dir = "/project/ll_774_951/InfoOpsNationwiseDriverControl"
countries_dir = os.listdir(
    "/project/ll_774_951/InfoOpsNationwiseDriverControl")

dataset_dirs = []
print("I am working")
for country in countries_dir:
    print(country)
    files_dir = os.listdir(os.path.join(root_dir, country))

    # Country File Names Check
    control_check = list(filter(lambda x: "control" in x, files_dir))
    treated_check = list(
        filter(lambda x: "tweets_csv_unhashed" in x, files_dir))

    if (len(control_check) >= 1 and len(treated_check) >= 1):
        dataset_dirs.append(os.path.join(root_dir, country, treated_check[0]))
        dataset_dirs.append(os.path.join(root_dir, country, control_check[0]))

GenerateDatasets(dataset_dirs)
