import json
import os
import sys

import pandas as pd
import networkx as nx

import warnings

# Script to be added
def generate_jsons(json_file_name,graph_dir):
    G = nx.read_gexf(graph_dir)
    nodes = list(set(G.nodes))
    jsons = []

    for node in nodes:
        temp_json = {"instruction":"Determine if the user is actively driving an influence campaign."}
        neigbors_ids = list(G.neighbors(node))
        neigbors_text = ','.join(neigbors_ids)
        temp_json["input"] = node  + " is connected to " + neigbors_text
        temp_json['user'] = str(node)
        jsons.append(temp_json)
    
    with open("{FILENAME}_json_resp.json".format(FILENAME=json_file_name),"w") as file:
        file.write(json.dumps(jsons))
    
    warnings.warn("Written the json output")



file_path = "C:/Users/Ashwin/OneDrive/Desktop/infoOps/coordinatedActivity/cache_files/coURL_INCAS_0908_eigen.gexf"
generate_jsons("courl",file_path)



    

