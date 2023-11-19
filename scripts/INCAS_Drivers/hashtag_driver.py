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

dataset_dir = "/scratch1/ashwinba/consolidated/INCAS"
graph_dir = "/scratch1/ashwinba/cache/INCAS"

with gzip.open(os.path.join(dataset_dir,"consolidated_INCAS_0908.csv.gz")) as f:
    cum_df = pd.read_csv(f)

warnings.warn("opened dataframe")
G = hashSeq(cum_df)

# Saving Graph in GML File
nx.write_gexf(G,os.path.join(graph_dir,"hashSeq_INCAS.gexf"))
warnings.warn("file written")


