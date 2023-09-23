import os
import pandas as pd
import numpy as np

import pickle
import networkx as nx
import gzip

from coordinatedActivity.hashtagSeq import *

root_dir = "/project/ll_774_951/InfoOpsNationwiseDriverControl"
dataset_dir = "/scratch1/ashwinba/consolidated"
graph_dir = "/scratch1/ashwinba/graphs"

treated_df = pd.read_csv(os.path.join(dataset_dir,"treated_consolidated_raw.csv.gz"),compression='gzip')
control_df = pd.read_csv(os.path.join(dataset_dir,"control_consolidated_raw.csv.gz"),compression='gzip')

G = hashSeq(control_df, treated_df)

# Saving Graph in GML File
nx.write_gml(G,os.path.join(graph_dir,"hashtagSeq.gml.gz"))


