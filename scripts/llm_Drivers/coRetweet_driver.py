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

from coordinatedActivity.INCAS_scripts.coRetweet_v3 import *

#Declare directories and file_name
dataset_dir = "/scratch1/ashwinba/data" # File Location
graph_dir = "/scratch1/ashwinba/cache/llms" #Final destination of graph
file_name = "df_test_russia.csv" # Name of the File to be read
country_name = file_name.split(".")[0]

warnings.warn(country_name)

try:
    with gzip.open(os.path.join(dataset_dir,file_name)) as f:
        cum_df = pd.read_csv(f)
except:
    cum_df = pd.read_csv(os.path.join(dataset_dir,file_name))

warnings.warn("opened dataframe")
G = coRetweet(cum_df)

# Removing self loops
G.remove_edges_from(nx.selfloop_edges(G))

# Saving Graph in GML File
nx.write_gexf(G,os.path.join(graph_dir,"coRetweet_llms_{COUNTRY}.gexf".format(COUNTRY=country_name)))
warnings.warn("file written")


