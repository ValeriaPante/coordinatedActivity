import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import networkx as nx

import pickle

from coordinatedActivity.v3_scripts.textSimilarity_v3 import *

OUTPUT_DIR = "/scratch1/ashwinba/graphs"
root_dir = "/project/ll_774_951/InfoOpsNationwiseDriverControl"
countries_dir = os.listdir("/project/ll_774_951/InfoOpsNationwiseDriverControl")
OUTPUT_GRAPH_DIR = "/scratch1/ashwinba/cache"
dataset_dirs = []

'''for country in countries_dir:
    print(country)
    files_dir = os.listdir(os.path.join(root_dir,country))

    ## Country File Names Check
    control_check = list(filter(lambda x:"control" in x,files_dir))
    treated_check = list(filter(lambda x:"tweets_csv_unhashed" in x,files_dir))
    
    if(len(control_check) >= 1 and len(treated_check) >= 1):
        dataset_dirs.append(os.path.join(root_dir,country,treated_check[0]))
        dataset_dirs.append(os.path.join(root_dir,country,control_check[0]))
'''
#textSim(dataset_dirs,OUTPUT_DIR)
g = getSimilarityNetwork(OUTPUT_DIR)
# Saving Graph in GML File
nx.write_gml(g,os.path.join(OUTPUT_GRAPH_DIR,"text_sim.gml.gz"))