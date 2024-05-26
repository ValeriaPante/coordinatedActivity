import os
import sys
import pandas as pd
import numpy as np

import pickle
import networkx as nx

import gzip
import warnings

def process_urls(df):
    # Creating a copy of urls
    df['urls_old'] = df['urls']

    # Exploding it to string
    df['urls'] = df['urls'].apply(lambda rec: list(map(lambda x: x['expanded_url'],rec)))
    
    return df

# Importing coordinatedActivity root directory
sys.path.append('/scratch1/ashwinba/coordinatedActivity/scripts')

from coordinatedActivity.INCAS_scripts.coURL_v3 import *

# Output Directory
OUTPUT_DIR = "/project/muric_789/ashwin/INCAS/outputs_cuba_082020"

# Declare directories and file_name
dataset_dir = "/project/muric_789/ashwin/INCAS/processed_data" # File Location
file_name = "cuba_082020_tweets_combined.pkl.gz" # Name of the File to be read

# Reading dataframe
df = pd.read_pickle(os.path.join(dataset_dir,file_name),compression='gzip')
cum_df = process_urls(df)

# Delete Cache
del df

warnings.warn("opened dataframe")
G = coURL(cum_df)


# Saving Graph in GML File
nx.write_gexf(G,os.path.join(OUTPUT_DIR,"coURL_INCAS_TA2_CUBA_min_3.gexf"))
warnings.warn("file written")