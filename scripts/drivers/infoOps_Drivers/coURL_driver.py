import os
import pandas as pd
import numpy as np

# Importing packages
import pickle
import networkx as nx
import gzip

# Importing coordinatedActivity root directory
import sys
sys.path.append('/scratch1/ashwinba/coordinatedActivity/scripts')

from coordinatedActivity.v3_scripts.coURL_v3 import *

dataset_dir = "/project/muric_789/ashwin/consolidated"
graph_dir = "/project/muric_789/ashwin/outputs"

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
#nx.write_gexf(G,os.path.join(graph_dir,"coURL.gexf"))
nx.write_gml(G,os.path.join(graph_dir,"coURL.gml.gz"))


