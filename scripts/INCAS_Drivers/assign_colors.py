import networkx as nx
import os

import warnings

GRAPH_SAVE_DIR = "C:/Users/Ashwin/OneDrive/Desktop/infoOps/coordinatedActivity/cache_files"

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

    nx.draw(G, node_color=colors, with_labels=True, font_color='white')
    
    # Saving graph
    nx.write_gexf(G,os.path.join(GRAPH_SAVE_DIR,basename+"_colored.gexf"))
    warnings.warn("Saved Updated network")

assign_colors(graph_dir="C:/Users/Ashwin/OneDrive/Desktop/infoOps/coordinatedActivity/cache_files/coURL_INCAS_0908_eigen.gexf")






    