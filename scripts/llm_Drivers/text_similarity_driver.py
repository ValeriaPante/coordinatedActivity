import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import networkx as nx

import pickle
import warnings
import sys

# Importing coordinatedActivity root directory
sys.path.append('/scratch1/ashwinba/coordinatedActivity/scripts')

from coordinatedActivity.INCAS_scripts.textSimilarity_v3 import *

OUTPUT_DIR = "/scratch1/ashwinba/cache/llms/thresholds"
# Declare directories and file_name
dataset_dir = "/scratch1/ashwinba/data" # File Location
graph_dir = "/scratch1/ashwinba/cache/llms" #Final destination of graph
file_name = "df_train_ecuador.csv" # Name of the File to be read
country_name = file_name.split(".")[0]

try:
    with gzip.open(os.path.join(dataset_dir,file_name)) as f:
        cum_df = pd.read_csv(f)
except:
    cum_df = pd.read_csv(os.path.join(dataset_dir,file_name))

warnings.warn("opened dataframe")

textSim(cum_df,OUTPUT_DIR)
g = getSimilarityNetwork(OUTPUT_DIR)
warnings.warn("Similarity Network Recieved")

# Saving Graph in GML File
nx.write_gexf(g,os.path.join(graph_dir,"textsim_llm_{COUNTRY}.gexf".format(COUNTRY=country_name)))
warnings.warn("Sim Network Written")