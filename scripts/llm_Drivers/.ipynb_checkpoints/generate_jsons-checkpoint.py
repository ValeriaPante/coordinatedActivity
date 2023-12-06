import json
import os
import sys

import pandas as pd
import networkx as nx

import warnings

# Script to be added
def generate_jsons(graph_dir,output_dir):
    G = nx.read_gexf(graph_dir)
    nodes = list(set(G.nodes))
    jsons = []
    filename = os.path.splitext(os.path.basename(graph_dir))[0]

    for node in nodes:
        temp_json = {"instruction": "Determine if the user is actively driving an influence campaign."}
        neigbors_ids = list(G.neighbors(node))
        neigbors_text = ','.join(neigbors_ids)
        temp_json["input"] = node  + " is connected to " + neigbors_text
        temp_json['userid'] = str(node)
        jsons.append(temp_json)

    jsons_df = pd.DataFrame(jsons)
    ref_df = pd.read_csv(reference_csv)[['userid','control']]
    ref_df['output'] = ref_df['control'].apply(lambda x:False if x == True else True)
    ref_df = ref_df[['userid','output']]

    df_merged = pd.merge([jsons_df,ref_df],on='userid')
    df_merged.to_json(os.path.join(output_dir,"{FILENAME}_json_resp.json".format(FILENAME=filename)))
    
    warnings.warn("Written the json output")

# Declaring Directories
file_dir = "/scratch1/ashwinba/cache/llms"
reference_csv = "/scratch1/ashwinba/data/df_train_russia.csv"
output_dir ="/scratch1/ashwinba/cache/llms/json_resps"
FILE_NAME = "textsim_llm_russia.gexf"
graph_file_path = os.path.join(file_dir,FILE_NAME)

generate_jsons(graph_file_path,output_dir)
warning.warn("Json Created Successfully")


    

