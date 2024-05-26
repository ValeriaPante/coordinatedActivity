import warnings
import os
import glob

import pandas as pd
import numpy as np

import networkx as nx

from sklearn.metrics import normalized_mutual_info_score

import matplotlib.pyplot as plt
import seaborn as sns

import json

#GRAPH_DIR = "/scratch1/ashwinba/cache/INCAS/filtered_networks/"
GRAPH_DIR = "/scratch1/ashwinba/cache/INCAS/phase_2"
JSON_DIR = "/scratch1/ashwinba/cache/INCAS/phase_2/output_jsons"
#JSON_DIR = "/scratch1/ashwinba/cache/INCAS/old_networks/output_jsons"

# Ref CSV -> Root processed CSV File
CSV_DIR = "/scratch1/ashwinba/consolidated/INCAS/phase_2/consolidated_INCAS_EVAL_2.csv.gz"
#CSV_DIR = "/scratch1/ashwinba/consolidated/INCAS/consolidated_INCAS_0908.csv.gz"

ref_df = pd.read_csv(CSV_DIR,compression='gzip')

warnings.warn("read the csv")

def plot_cdf_curve(values,method):
    sns.set(font_scale=1.5, rc={'axes.facecolor':'white', 'figure.facecolor':'white'})
    sns.set_style("whitegrid")
    plt.figure()
    sns.kdeplot(data = values, cumulative = True,linewidth=5)
    percentile = np.percentile(values,90)
    warnings.warn("90th perentile : " + str(percentile))
    users_count = len(list(filter(lambda x:x>=percentile,values)))
    plt.axvline(x = percentile, color = 'g', linestyle='-',linewidth=2,label='97.5th pct. : ' + str(users_count) + " users")
    percentile = np.percentile(values,99.50)
    warnings.warn("99.5th percntile : " + str(percentile))
    users_count = len(list(filter(lambda x:x>=percentile,values)))
    plt.axvline(x = percentile, color = 'black', linestyle='-',linewidth=2,label='99.5th pct. : ' + str(users_count) + " users")

    plt.ylabel("density")
    plt.xlabel("eigen-centrality")
    plt.title(method)
    plt.legend(loc='best')
    plt.savefig(os.path.join(JSON_DIR,"{METHOD}_CDF.png".format(METHOD=method)))
    plt.show()


def filter_graphs(graphs_dir,threshold):
    graphs_dicts = {}
    filtered_dict = {}
    for graph in graphs_dir:
        method_name = os.path.basename(graph)
        
        # Reading the graph
        G = nx.read_gexf(graph)
        
        # Compute eigen centrality
        eigen_centrality = nx.eigenvector_centrality(G)
        
        # Calculate the percentile
        percentile = np.percentile(list(eigen_centrality.values()),threshold)
        
        warnings.warn(str(method_name))
        warnings.warn(str(percentile))
        
        # Filtering nodes
        filtered_nodes = list(dict(filter(lambda item: item[1]<percentile,eigen_centrality.items())).keys())

        # Deleting Eigen Centrality Object
        del eigen_centrality

        # Filtering operation
        G.remove_nodes_from(filtered_nodes)
        
        # Writing network file
        nx.write_gexf(G,os.path.join(GRAPH_DIR,"{METHOD}_{THRESH}.gexf".format(METHOD=method_name,THRESH=str(threshold))))
        
        # Deleting cache
        del G

        # Sample Warnings
        warnings.warn(str(method_name)+" done")

def generate_segments(users_dict,method_name,threshold,ref_df):
    
    # Naming conventions
    conventions_template = {'0':
        {"confidence":0,"description":"non coordinated users based on {METHOD}","name":"segment_non_coordinated_users_{METHOD}","text":"actors:{COUNT}"},
        '1':{"confidence":1,"description":"coordinated users based on {METHOD}","name":"segment_coordinated_users_{METHOD}","text":"actors:{COUNT}"}}
        
    json_items = []
    
    for key,users in users_dict.items():
        # IF the network is constructed using author
        filtered_df = ref_df[ref_df['author'].isin(users)]
        
        # If the network is constructed using userid -> constructed numerical convertive
        if(len(filtered_df) == 0):
            warnings.warn("Hello")
            ref_df['userid'] = ref_df['userid'].astype(str)
            filtered_df = ref_df[ref_df['userid'].isin(users)]

        template = conventions_template[key].copy()
        
        if("hash" in method_name and key == '1'):
            hash_tags  = set()
            for tweet in filtered_df['contentText'].values.astype(str):
                tags = list(set(["#"+str(tag.strip(".").strip("#").strip(":")) for tag in tweet.split() if tag.startswith("#")]))
                tags = list(set(filter(lambda x:x!=" " and len(x.strip("#"))>=3,tags)))
                if(len(tags) >=3):
                    hash_tags.add(str(tags))
            hash_tags = list(hash_tags)
            template['description'] = ','.join(hash_tags)
        
        elif("coURL" in method_name and key == '1'):
            filtered_df_1 = filtered_df.copy()
            filtered_df_1['urls'] = filtered_df_1['embeddedUrls'].astype(str).replace('[]', '').apply(lambda x: x[1:-1].replace("'", '').split(',') if len(x) != 0 else '')
            filtered_df_1 = filtered_df_1.loc[filtered_df_1['urls'] != ''].explode('urls')
            template['description'] = ','.join(set(filtered_df_1['urls'].values))
            del filtered_df_1
   
        else:
            template['description'] = template['description'].format(METHOD = method_name)
            
        authors = list(set(filtered_df['author'].values.tolist()))
        
        template['name'] = template['name'].format(METHOD=method_name.split("_")[0])
        template['text'] = template['text'].format(COUNT=str(len(authors)))
        
        template['actors'] = authors
        template['providerName'] = "ta2-usc"
        
        json_items.append(template)
        del template
        del filtered_df
        
    json_to_be_saved = {"segments":json_items}
    
    with open(os.path.join(JSON_DIR,"segments_{METHOD}_THRESH_{THRESHOLD}.json".format(METHOD=method_name,THRESHOLD=str(threshold))), 'w+', encoding ='utf8') as json_file: 
        json.dump(json_to_be_saved, json_file, indent = 6,ensure_ascii=False) 
    
    json_file.close()  

    del json_to_be_saved
    

    
def generate_messages(users_dict,method_name,threshold,ref_df):
    
    # Naming conventions
    conventions_template = {'0':
        {"confidence":0,"description":"messages from non coordinated users based on {METHOD} indicator","name":"messagegroup_non_coordinated_users_{METHOD}","text":"messages:{COUNT}"},
        '1':{"confidence":1,"description":"messages from coordinated users based on {METHOD} indicator","name":"messagegroup_coordinated_users_{METHOD}","text":"messages:{COUNT}"}}

    json_items = []
    
    if("textsim" in method_name):
        ref_df = ref_df[ref_df['engagementType']!='retweet']

    
    for key,users in users_dict.items():
        
        
        template = conventions_template[key].copy()

        
        # IF the network is constructed using author
        filtered_df = ref_df[ref_df['author'].isin(users)]
        
        # If the network is constructed using userid -> constructed numerical convertive
        if(len(filtered_df) == 0):
            ref_df['userid'] = ref_df['userid'].astype(str)
            filtered_df = ref_df[ref_df['userid'].isin(users)]
            
        if("coRetweet" in method_name):
            filtered_df.dropna(subset=['retweet_id'],inplace=True)
            filtered_df['retweet_id'] = filtered_df['retweet_id'].astype(str)
            template['messages'] = filtered_df['retweet_id'].values.tolist()
            
        else:
            template['messages'] = filtered_df['tweetid'].values.tolist()
            
        
        template['type'] = 'tweet ID'
        template['providerName'] = 'ta2-usc'
        
        template['description'] = template['description'].format(METHOD=method_name.split("_")[0])
        template['name'] = template['name'].format(METHOD=method_name)
        template['text'] = template['text'].format(COUNT=str(len(filtered_df)))
        
        json_items.append(template)
        del template

    json_to_be_saved = {"messagegroups":json_items}
    
    with open(os.path.join(JSON_DIR,"message_groups_{METHOD}_THRESH_{THRESHOLD}.json".format(METHOD=method_name,THRESHOLD=str(threshold))), 'w+', encoding ='utf8') as json_file: 
        json.dump(json_to_be_saved, json_file, indent = 6,ensure_ascii=False) 
    
     
    json_file.close()  

def retrieve_users(graphs_dir,thresholds,ref_df):
    
    for graph_dir in graphs_dir:

        method_name = os.path.basename(graph_dir)
        
        # Reading the graph
        G = nx.read_gexf(graph_dir)

        G.remove_edges_from(nx.selfloop_edges(G))
        G.remove_nodes_from(list(nx.isolates(G)))

        # Compute eigen centrality
        eigen_centrality = nx.eigenvector_centrality(G)
        
        for threshold in thresholds:
        
            # Calculate the percentile
            percentile = np.percentile(list(eigen_centrality.values()),threshold)
    
            # Users 
            users = list(G.nodes)
            
            # filter_users
            coordinated_users = list(set(list(dict(filter(lambda item: item[1]>=percentile,eigen_centrality.items())).keys())))
            non_coordinated_users = list(set(list(filter(lambda x:x not in coordinated_users,users))))
            
            # Generate_JSONs
            generate_segments({'1':coordinated_users,'0':non_coordinated_users},method_name,threshold,ref_df)
            #generate_messages({'1':coordinated_users,'0':non_coordinated_users},method_name,threshold,ref_df)
            
            # Test warning
            warnings.warn("Completed for {METHOD} - {THRESH}".format(METHOD=method_name,THRESH=str(threshold)))
            
ROOT_DIR = "/scratch1/ashwinba/cache/INCAS/phase_2/fused_network"
# ROOT_DIR = "/scratch1/ashwinba/cache/INCAS/old_networks"
graphs_dir = glob.glob(os.path.join(ROOT_DIR,"*.gexf"))
#graphs_dir = [os.path.join(ROOT_DIR,"hashSeq_INCAS_TA2_min_hashtags_3.gexf")]
#graphs_dir = [os.path.join(ROOT_DIR,"coURL_INCAS_TA2_1_V1.gexf")]
#graphs_dir = ["/scratch1/ashwinba/cache/INCAS/phase_2/hashSeq_INCAS_TA2_minhash_2.gexf"]
#graphs_dir = [os.path.join(ROOT_DIR,"coRetweet_INCAS.gexf")]
warnings.warn(str(graphs_dir))

# thresholds = [i for i in range(50,100,2)]
# for thresh in thresholds:
#     filter_graphs(graphs_dir=graphs_dir,threshold=thresh)

#retrieve_users(graphs_dir,thresholds=[85],ref_df=ref_df)
#retrieve_users(graphs_dir,thresholds = [90],ref_df=ref_df)
retrieve_users(graphs_dir,thresholds = [95,98],ref_df=ref_df)


