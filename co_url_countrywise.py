import os
import pandas as pd

import networkx as nx
import matplotlib.pyplot as plt



def generate_country_wise(mapper1,mapper2,graph):
    graph.remove_edges_from(list(nx.selfloop_edges(G)))

    # initialize graph
    G1 = nx.Graph()
    countries = list(set(list(mapper1['country'].values) + list(mapper2['country'].values)))

    # Generate Vertices
    graph = nx.relabel_nodes(graph, lambda x: str(int(eval(x.strip()))))
    
    #print(mapper1.head(5))
    #print(mapper2.head(5))

    # Initialize Vertices
    for country in countries:
        G1.add_node(country)    


    vertices = list(set(graph.nodes))
    for v in vertices:
        vertice = str(int(eval(v.strip())))
        # Generate Country Wise
        if(vertice in list(mapper1['userid'].values) or vertice in list(mapper2['userid'].values)):
            try:
                from_c = list(mapper1[mapper1['userid'] == vertice]['country'].values)[0]
            except IndexError:
                from_c = list(mapper2[mapper2['userid'] == vertice]['country'].values)[0]
            except:
                pass
            nodes = list(nx.node_connected_component(graph,vertice))
            
            for n1 in nodes:
                node = str(int(eval(n1.strip())))
                # Generate Country Wise
                if(node in list(mapper1['userid'].values) or node in list(mapper2['userid'].values)):
                    try:
                        to_c = list(mapper1[mapper1['userid'] == node]['country'].values)[0]
                    except IndexError:
                        to_c = list(mapper2[mapper2['userid'] == node]['country'].values)[0]
                    except:
                        pass
                    
                    if from_c!=to_c:
                        if G1.has_edge(from_c,to_c):
                                    # we added this one before, just increase the weight by one
                                    G1[from_c][to_c]['weight'] += 1
                        else:
                            # new edge. add with weight=1
                            G1.add_edge(from_c,to_c, weight=1)
                else:pass
        
    # Save
    nx.draw(G1, with_labels=True, font_weight='normal')
    plt.savefig("/scratch1/ashwinba/cache/plt_countrywise_courl_img.png")

    print("Executed")

    
RAW_GRAPH_DIR = "/scratch1/ashwinba/cache/co_url.gml.gz"
MAPPER_DIR = "/scratch1/ashwinba/consolidated"
CONTROL_MAPPER_DIR = "control_consolidated_ids.csv.gz"
TREATED_MAPPER_DIR = "treated_consolidated_ids.csv.gz"

G = nx.read_gml(os.path.join(RAW_GRAPH_DIR))
mapper1 = pd.read_csv(os.path.join(MAPPER_DIR,CONTROL_MAPPER_DIR),compression='gzip',low_memory=False)
mapper2 = pd.read_csv(os.path.join(MAPPER_DIR,TREATED_MAPPER_DIR),compression='gzip',low_memory=False)

# Calling Function
generate_country_wise(mapper1,mapper2,G)
