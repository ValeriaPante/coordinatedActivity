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

# Output Directory
OUTPUT_DIR = "/project/muric_789/ashwin/INCAS/outputs_cuba_082020"

# Declare directories and file_name
dataset_dir = "/project/muric_789/ashwin/INCAS/processed_data" # File Location
file_name = "cuba_082020_tweets_combined.pkl.gz" # Name of the File to be read

cum_df = pd.read_pickle(os.path.join(dataset_dir,file_name),compression='gzip')

warnings.warn("opened dataframe")
G = hashSeq(cum_df)

# Saving Graph in GML File
nx.write_gexf(G,os.path.join(OUTPUT_DIR,"hashSeq_INCAS_TA2_CUBA.gexf"))
warnings.warn("file written")