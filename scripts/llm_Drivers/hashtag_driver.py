import os
import sys
import pandas as pd
import numpy as np

import pickle
import networkx as nx

import gzip
import warnings

# Importing coordinatedActivity root directory
sys.path.append('/scratch1/ashwinba/coordinatedActivity/scripts')

from coordinatedActivity.INCAS_scripts.hashtagSeq_v3 import *

# Declare directories and file_name
dataset_dir = "/scratch1/ashwinba/data" # File Location
graph_dir = "/scratch1/ashwinba/cache/llms" #Final destination of graph
file_name = "df_train_russia.csv" # Name of the File to be read
country_name = file_name.split("_")[-1].split(".")[0]

try:
    with gzip.open(os.path.join(dataset_dir,file_name)) as f:
        cum_df = pd.read_csv(f)
except:
    cum_df = pd.read_csv(os.path.join(dataset_dir,file_name))
    
print(cum_df.shape)

warnings.warn("opened dataframe")
G = hashSeq(cum_df)

# Saving Graph in GML File
nx.write_gexf(G,os.path.join(graph_dir,"hashSeq_llms{COUNTRY}.gexf".format(COUNTRY=country_name)))
warnings.warn("file written")

