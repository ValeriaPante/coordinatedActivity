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

OUTPUT_DIR = "/scratch1/ashwinba/cache/INCAS/phase_2/graphs"
# Declare directories and file_name
dataset_dir = "/scratch1/ashwinba/consolidated/INCAS/phase_2" # File Location
graph_dir = "/scratch1/ashwinba/cache/INCAS/phase_2" #Final destination of graph
file_name = "consolidated_INCAS_EVAL_2.csv.gz" # Name of the File to be read

# Phase-2 Directory
# dataset_dir = "/scratch1/ashwinba/consolidated/INCAS/phase_2"
# graph_dir = "/scratch1/ashwinba/cache/INCAS/phase_2"
# file_name = "processed_INCAS_TA2.csv.gz"

with gzip.open(os.path.join(dataset_dir,file_name)) as f:
    cum_df = pd.read_csv(f)

warnings.warn("opened dataframe")
G = hashSeq(cum_df)

# Saving Graph in GML File
nx.write_gexf(G,os.path.join(graph_dir,"hashSeq_INCAS_TA2.gexf"))
warnings.warn("file written")


