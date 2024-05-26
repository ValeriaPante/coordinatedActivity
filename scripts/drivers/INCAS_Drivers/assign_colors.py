import networkx as nx
import os
import sys
import pandas as pd

import warnings

GRAPH_SAVE_DIR = "/scratch1/ashwinba/cache/INCAS"

def set_color(attrs,node,threshold):
    # Color Mappings
    color_mappings = {"treated":"red","control":"blue"}
    eigen_centrality = attrs[node]
    if(threshold>eigen_centrality):
        return color_mappings["treated"]
    
    # Return red color node for control users
    return color_mappings["control"]

## Driver code
def assign_colors(graph_dir,threshold=0.9):
    G = nx.read_gexf(graph_dir)
    warnings.warn("Read graph file")

    # Retrieving the filename
    basename = os.path.basename(graph_dir).replace(".gexf","")

    # Eigen Attributes
    # title="eigen_centrality"
    attrs = nx.get_node_attributes(G,"eigen_centrality")
    colors =  list(map(lambda x:set_color(attrs,x,threshold),list(attrs.keys())))
    warnings.warn("Computed colors")

    
    # Saving Temp Node File
    nodes = list(attrs.keys())
    eigen_centrality= list(attrs.values())
    attr_csv = pd.DataFrame(zip(nodes,eigen_centrality,colors),columns=['node_id','eigen_centrality','color_mappings'])
    attr_csv['label'] = attr_csv['color_mappings'].apply(lambda x:"treated" if x=="red" else "control")
    attr_csv.to_csv(os.path.join(GRAPH_SAVE_DIR,basename+"_eigen_node_mappings.csv"))
    warnings.warn("Saved_eigen_node_mappings csv")
    
    
    nx.draw(G, node_color=colors, with_labels=True, font_color='white')
    
    # Saving graph
    nx.write_gexf(G,os.path.join(GRAPH_SAVE_DIR,basename+"_colored.gexf"))

    
    warnings.warn("Saved Updated network")



if __name__ == "__main__":
    graph_dir = sys.argv[1]
    assign_colors(graph_dir = graph_dir)






    