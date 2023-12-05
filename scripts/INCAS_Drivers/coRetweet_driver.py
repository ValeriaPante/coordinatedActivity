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

# Declare directories and file_name
dataset_dir = "/scratch1/ashwinba/consolidated/INCAS" # File Location
graph_dir = "/scratch1/ashwinba/cache/INCAS" #Final destination of graph
file_name = "consolidated_INCAS_0908.csv.gz" # Name of the File to be read

with gzip.open(os.path.join(dataset_dir,file_name)) as f:
    cum_df = pd.read_csv(f)

warnings.warn("opened dataframe")
G = coRetweet(cum_df)

# Saving Graph in GML File
nx.write_gexf(G,os.path.join(graph_dir,"coRetweet_INCAS_0908.gexf"))
warnings.warn("file written")


