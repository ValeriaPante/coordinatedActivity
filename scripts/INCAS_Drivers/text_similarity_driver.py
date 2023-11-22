import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import networkx as nx

import pickle
import warnings

from coordinatedActivity.INCAS_scripts.textSimilarity_v3 import *

OUTPUT_DIR = "/scratch1/ashwinba/cache/INCAS/graphs"
OUTPUT_GRAPH_DIR = "/scratch1/ashwinba/cache/INCAS"
dataset_dir = "/scratch1/ashwinba/consolidated/INCAS"
graph_dir = "/scratch1/ashwinba/cache/INCAS"

with gzip.open(os.path.join(dataset_dir,"consolidated_INCAS_0908.csv.gz")) as f:
    cum_df = pd.read_csv(f)

warnings.warn("opened dataframe")

textSim(cum_df,OUTPUT_DIR)
g = getSimilarityNetwork(OUTPUT_DIR)
warnings.warn("Similarity Network Recieved")

# Saving Graph in GML File
nx.write_gexf(g,os.path.join(graph_dir,"textsim_INCAS.gexf"))
warnings.warn("Sim Network Written")