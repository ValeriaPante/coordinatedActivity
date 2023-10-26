import os
import pandas as pd
import numpy as np

import pickle
import networkx as nx

import gzip

from coordinatedActivity.v3_scripts.coRetweet_v3 import *

root_dir = "/project/ll_774_951/InfoOpsNationwiseDriverControl"
dataset_dir = "/scratch1/ashwinba/consolidated"
graph_dir = "/scratch1/ashwinba/cache"

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
nx.write_gml(G,os.path.join(graph_dir,"coRetweet.gml.gz"))
print("Done")


