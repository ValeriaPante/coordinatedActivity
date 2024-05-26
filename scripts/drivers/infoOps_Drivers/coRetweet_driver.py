import os
import pandas as pd
import numpy as np

import pickle
import networkx as nx

import gzip

# Importing coordinatedActivity root directory
import sys
sys.path.append('/scratch1/ashwinba/coordinatedActivity/scripts')

from coordinatedActivity.v3_scripts.coRetweet_v3 import *

dataset_dir = "/project/muric_789/ashwin/consolidated"
graph_dir = "/project/muric_789/ashwin/outputs"

with gzip.open(os.path.join(dataset_dir,"treated_consolidated_raw.csv.gz")) as f:
    treated_df = pd.read_csv(f)

with gzip.open(os.path.join(dataset_dir,"control_consolidated_raw.csv.gz")) as f:
    control_df = pd.read_csv(f)

#treated_df = pd.read_csv(os.path.join(dataset_dir,"treated_consolidated_raw.csv.gz"),compression='gzip',engine="python",index_col=None)
#control_df = pd.read_csv(os.path.join(dataset_dir,"control_consolidated_raw.csv.gz"),compression='gzip',engine="python",index_col=None)

print(len(control_df['country'].unique()))
print(len(treated_df['country'].unique()))

control_df1 = control_df[['user', 'retweeted_status', 'id']]
treated_df1 = treated_df[['tweetid', 'userid', 'retweet_tweetid']]

del treated_df
del control_df

G = coRetweet(control_df1, treated_df1)

# Saving Graph in GML File
nx.write_gexf(G,os.path.join(graph_dir,"coRetweet.gexf"))
nx.write_gml(G,os.path.join(graph_dir,"coRetweet.gml.gz"))
print("Done")


