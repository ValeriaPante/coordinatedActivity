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

from coordinatedActivity.INCAS_scripts.coURL_v3 import *

dataset_dir = "/scratch1/ashwinba/consolidated/INCAS"
graph_dir = "/scratch1/ashwinba/cache/INCAS"

with gzip.open(os.path.join(dataset_dir,"consolidated_INCAS.csv.gz")) as f:
    cum_df = pd.read_csv(f)

warnings.warn("opened dataframe")
G = coURL(cum_df)

# Saving Graph in GML File
nx.write_gml(G,os.path.join(graph_dir,"coURL_INCAS.gml.gz"))
warnings.warn("file written")


