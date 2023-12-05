import json
import os
import sys

import pandas as pd
import networkx as nx

import warnings

# Script to be added
def genearate_jsons(json_file_name,graph_dir):
    G = nx.read_gexf(graph_dir)
    nodes = list(set(G.nodes))
    jsons = []

    for node in nodes:
        temp_json = {"instruction":"Determine if the user is actively driving an influence campaign."}
        neigbors_ids = [G[temp_node]["id"] for temp_node in G.neigbours(node)]
        neigbors_text = ','.join(neigbors_ids)
        temp_json["input"] = node  + "is connected to " + neigbors_text
        jsons.append(temp_json)
    
    with open("{FILENAME}_json_resp.json","w") as file:
        file.write(jsons)
    
    warnings.warn("Written the json output")





    

