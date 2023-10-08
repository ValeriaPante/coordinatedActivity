import os
import pandas as pd
import numpy as np

import pickle
import networkx as nx
import gzip

from coordinatedActivity.v3_scripts.coURL_v3 import *

root_dir = "/project/ll_774_951/InfoOpsNationwiseDriverControl"
dataset_dir = "/scratch1/ashwinba/consolidated"
graph_dir = "/scratch1/ashwinba/graphs"

treated_df = pd.read_csv(os.path.join(dataset_dir,"treated_consolidated_raw.csv.gz"),compression='gzip')
control_df = pd.read_csv(os.path.join(dataset_dir,"control_consolidated_raw.csv.gz"),compression='gzip')

#['user', 'entities', 'id']
#     - treated: information Operation dataset -> includes only columns ['tweetid', 'userid', 'urls']

print(len(control_df['country'].unique()))
print(len(treated_df['country'].unique()))


control_df = control_df[['user', 'entities', 'id']]
treated_df = treated_df[['tweetid', 'userid', 'urls']]
G = coRetweet(control_df, treated_df)

# Saving Graph in GML File
nx.write_gml(G,os.path.join(graph_dir,"coURL.gml.gz"))


