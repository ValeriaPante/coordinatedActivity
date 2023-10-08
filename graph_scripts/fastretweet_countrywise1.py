import os
import pandas as pd

import networkx as nx
import matplotlib.pyplot as plt

import random
import warnings

def generate_country_wise(mapper1,mapper2,graph):
    # Generate Vertices
    graph = nx.relabel_nodes(graph, lambda x: str(int(eval(x.strip()))))
    graph.remove_edges_from(list(nx.selfloop_edges(graph)))
    vertices = list(set(graph.nodes))
    warnings.warn("Total Vertices :"+str(len(vertices)))
    
    mapper1['class_label'] = "control"
    mapper2['class_label'] = "treated"
    
    #countries = list(set(list(set(list(mapper1['country'].values) + list(mapper2['country'].values)))))

    merged = pd.concat([mapper1,mapper2])
    
    #Delete NaN Values
    merged.dropna(subset=['userid'],inplace=True)
    
    merged['userid'] = merged['userid'].apply(lambda x: str(x))
    warnings.warn("userid dtype changed")
    
    # Deleting Consolidated Files
    del mapper1
    del mapper2
    
    attributes = {}
    counter = 0
    warnings.warn("Starting to work")
    for v in vertices:
        counter+=1
        if(counter%10 == 0):
            warnings.warn(str(counter) + " work done")
        temp_dict = {}
        
        if((merged['userid'] == v).any()):
            temp_dict['class_label'] = merged[merged.userid == v].iloc[0]['class_label']

        else:
            warnings.warn(str(v)+" not found")
            temp_dict["class_label"] = "treated"
        
        temp_dict["userid"] = str(v)
        country_mapped = merged.loc[merged['userid'] == v,'country']
        if(len(country_mapped) == 0 or country_mapped.values[0] in ["egypt","uae"]):
            temp_dict["country"] = "Egypt&UAE"
        else:
            temp_dict["country"] = country_mapped.values[0]

        attributes[v] = temp_dict
            
  
    # Save
    nx.set_node_attributes(graph,attributes)
    nx.draw(graph, with_labels=True, font_weight='normal')
    nx.write_gexf(graph,"/scratch1/ashwinba/cache/plt_countrywise_fastretweet.gexf")
    #plt.savefig("/scratch1/ashwinba/cache/plt_countrywise_fastretweet_img.png")

    print("Executed")

    
RAW_GRAPH_DIR = "/scratch1/ashwinba/cache/fast_retweet.gml.gz"
MAPPER_DIR = "/scratch1/ashwinba/consolidated"
CONTROL_MAPPER_DIR = "control_consolidated_raw.csv.gz"
TREATED_MAPPER_DIR = "treated_consolidated_raw.csv.gz"

G = nx.read_gml(os.path.join(RAW_GRAPH_DIR))
control = pd.read_csv(os.path.join(MAPPER_DIR,CONTROL_MAPPER_DIR),compression='gzip')
treated = pd.read_csv(os.path.join(MAPPER_DIR,TREATED_MAPPER_DIR),compression='gzip')

# Dropping NaN Values
control.dropna(subset=['user','country'],inplace=True)
#treated.dropna(subset=['userid','country'],inplace=True)

control['userid'] = control['user'].apply(lambda x: str(int(eval(x)['id'].strip())))

mapper1 = control[['country','userid']]
mapper2 = treated[['country','userid']]

# Deleting dfs
del control
del treated

#mapper1['userid'] = mapper1['userid'].astype(str)
#mapper2['userid'] = mapper2['userid'].astype(str)

# Calling Function
generate_country_wise(mapper1,mapper2,G)