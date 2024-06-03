import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import networkx as nx

import pickle
import warnings
import sys

# Importing coordinatedActivity root directory
sys.path.append('/scratch1/ashwinba/scripts/coordinatedActivity/scripts')

from coordinatedActivity.INCAS_scripts.textSimilarity_v3 import *

OUTPUT_DIR = "/scratch1/ashwinba/data/temp/text_sim"
# Declare directories and file_name
dataset_dir = "/project/muric_789/ashwin/INCAS/processed_data" # File Location
graph_dir = "/scratch1/ashwinba/data/INCAS/EVAL_2B_SAMPLE/indicators" #Final destination of graph
file_name = "consolidated_INCAS_NEW_EVAL_2.csv.gz" # Name of the File to be read

with gzip.open(os.path.join(dataset_dir,file_name)) as f:
    cum_df = pd.read_csv(f)

warnings.warn("opened dataframe")
#textSim(cum_df,OUTPUT_DIR)
agg = 'max'
g = getSimilarityNetwork(OUTPUT_DIR,agg=agg)
warnings.warn("Similarity Network Recieved")

# Saving Graph in GML File
nx.write_gexf(g,os.path.join(graph_dir,f'textsim_INCAS_T2_V1_{agg}.gexf'))
warnings.warn("Sim Network Written")