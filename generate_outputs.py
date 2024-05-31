import pandas as pd
import networkx as nx

import numpy as np
import os

import json

def generate_outputs_pairs(output_dir,graphs,data_dir):
    df = pd.read_csv(data_dir,compression='gzip')
    total_edges = []

    for graph,thresh in graphs:
        G = nx.read_gexf(graph)
        G.remove_edges_from(nx.selfloop_edges(G))

        # Fetch Edges
        edges = list(G.edges.data())
        edge_weights = list(map(lambda x:float(x[2]['weight']), edges))

        percentile_value = np.percentile(edge_weights,thresh)
        edges_1 = list(filter(lambda x:x[2]['weight']>=percentile_value,edges))
        edges_1 = [edge[:2] for edge in edges_1]
        print(edges_1[0])
        
        print("edges",len(set(edges_1)))
        total_edges+=edges_1

    coordinated_users_lst = np.unique(total_edges)
    
    coordinated_users = df[df['userid'].astype(str).isin(coordinated_users_lst)][['userid','author']]
    non_coordinated_users = df[~df['userid'].astype(str).isin(coordinated_users_lst)]['author'].unique().tolist()

    hmap = dict(
        zip(
            coordinated_users['userid'].astype(str).values.tolist(),coordinated_users['author'].values.tolist()
        ))
    
    unique_edges = []
    for edge in total_edges:
        a,b  = hmap[edge[0]],hmap[edge[1]]
        if((a,b) not in total_edges) and ((b,a) not in total_edges):
            unique_edges.append((a,b))
    
    coordinated = {
        'confidence':1,
        'description':"coordinated pairs based on unified indicator",
        'name':"coordinated users pairs",
        'pairs':unique_edges
    }
    coordinated['text'] = f'edges:{len(unique_edges)},users:{len(coordinated_users_lst)}'

    non_coordinated = {
        'confidence':0,
        'description':"non coordinated users based on unified indicator",
        'name':"non coordinated users",
        'text':f'users:{len(non_coordinated_users)}',
        'actors':non_coordinated_users
    }
    users = [coordinated,non_coordinated]

    # with open(os.path.join(output_dir,"segments.json"),"w+") as f:
    #     json.dump({"segments":users},f)

    return users,hmap

#data_dir = "/project/muric_789/ashwin/INCAS/processed_data/consolidated_INCAS_NEW_EVAL_2.csv.gz"
# graphs_dirs = [
#     ["/scratch1/ashwinba/data/INCAS/EVAL_2B/indicators/hashSeq_INCAS_TA2_minhash_2.gexf",90],
#     ["/scratch1/ashwinba/data/INCAS/EVAL_2B/indicators/textsim_INCAS_T2_V1.gexf",99],
#     ["/scratch1/ashwinba/data/INCAS/EVAL_2B/indicators/coRetweet_INCAS_TA2.gexf",99.99],
# ]