import os
import pandas as pd
import numpy as np

import pickle
import networkx as nx
import gzip

# Importing coordinatedActivity root directory
import sys
sys.path.append('/scratch1/ashwinba/coordinatedActivity/scripts')

from coordinatedActivity.v3_scripts.hashtagSeq_v3 import *

dataset_dir = "/project/muric_789/ashwin/consolidated"
graph_dir = "/project/muric_789/ashwin/outputs"

treated_df = pd.read_csv(os.path.join(dataset_dir,"treated_consolidated_raw.csv.gz"),compression='gzip')
control_df = pd.read_csv(os.path.join(dataset_dir,"control_consolidated_raw.csv.gz"),compression='gzip')

control = control_df[['retweeted_status', 'user', 'in_reply_to_status_id', 'tweet_text', 'id']]
treated = treated_df[['retweet_tweetid', 'userid', 'in_reply_to_tweetid', 'quoted_tweet_tweetid', 'tweet_text', 'tweetid']]

# Renaming columns for convention
control.rename(columns = {'tweet_text':'full_text'}, inplace = True)

del treated_df
del control_df

G = hashSeq(control, treated)

# Saving Graph in GML File
#nx.write_gexf(G,os.path.join(graph_dir,"hashtagSeq.gexf"))
nx.write_gml(G,os.path.join(graph_dir,"hashtagSeq.gml.gz"))


