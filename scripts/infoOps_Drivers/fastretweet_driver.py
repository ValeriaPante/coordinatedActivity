import os
import pandas as pd
import numpy as np

import pickle
import networkx as nx
import gzip

from coordinatedActivity.v3_scripts.fastRetweet_v3 import *

root_dir = "/project/ll_774_951/InfoOpsNationwiseDriverControl"
dataset_dir = "/scratch1/ashwinba/consolidated"
graph_dir = "/scratch1/ashwinba/graphs"

treated_df = pd.read_csv(os.path.join(dataset_dir,"treated_consolidated_raw.csv.gz"),compression='gzip')
control_df = pd.read_csv(os.path.join(dataset_dir,"control_consolidated_raw.csv.gz"),compression='gzip')

control_df = control_df[['user', 'retweeted_status', 'id']]
treated_df = treated_df[['tweetid', 'userid', 'retweet_tweetid', 'retweet_userid']]

G = fastRetweet(control_df, treated_df)

# Saving Graph in GML File
nx.write_gml(G,os.path.join(graph_dir,"fast_retweet.gml.gz"))

