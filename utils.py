import pandas as pd
import numpy as np
import networkx as nx

from graspologic.embed import node2vec_embed

from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

def mergeNetworks(singleFeatureNets, weighted=False):
    """
    Merges multiple networks in a single one, where two nodes will be connected if they are in at least one of the input networks.

    Args:
        singleFeatureNets: List of networks to merge, each network must be a networkx.Graph object
        weighted: boolean variable indicating if the merged network should me weighted or not. If True, multiple weights for the same edge are grouped taking the maximum.
    Returns:
        M: merged network
    """

    graphs = []

    for net in singleFeatureNets:
        if weighted:
            df = pd.DataFrame(net.edges(data='weight'))
        else:
            df = pd.DataFrame(net.edges())
        graphs.append(df)

    temp = pd.concat([df for df in graphs])
    temp = temp.loc[temp[0]!=temp[1]]
    
    if weighted:
        temp.columns = ['source', 'target', 'weight']
        temp = temp.groupby(['source', 'target'], as_index=False).max()
    else:
        temp.columns = ['source', 'target']
        
    temp.dropna(inplace=True)

    if weighted:
        M = nx.from_pandas_edgelist(temp, edge_attr=True)
    else:
        M = nx.from_pandas_edgelist(temp)
        
    return M

def computeCentrality(graphPath):
    """
    Computes the eigenvector centrality for every node in a given network.

    Args:
        graphPath: path of the graph file in .gexf extension
    Returns:
        df: a DataFrame collecting the eigenvector centrality for every user (columns= [userid, eigenvectorCentr])
    """
    G = nx.read_gexf(graphPath)
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    G.remove_nodes_from(list(nx.isolates(G)))
    
    eigenvector_dict = nx.eigenvector_centrality(G, max_iter=500)
    nx.set_node_attributes(graph, eigenvector_dict, 'eigenvectorCentr')
    
    df = pd.DataFrame(graph.nodes(data='eigenvectorCentr'))
    df.columns = ['userid', 'eigenvectorCentr']
    df['userid'] = df['userid'].astype(str)

    return df

def getNodeEmbeddings(input_file, output_file):
    """
    Computes the node embeddings (dim = 128) of each node in a given network.

    Args:
        input_file: path of the graph file in .gexf extension
        output_file: path where to save the output file
    Returns:

    """
    
    G = nx.read_gexf(input_file)
    G = nx.Graph(G)

    node2vec_embedding, node_labels = node2vec_embed(
        G, num_walks=16, walk_length=16, workers=16, inout_hyperparameter=1.0, return_hyperparameter=1.0
    )
    
    np.save(output_file+'_embed.npy', node2vec_embedding) # saves the emebddings
    np.save(output_file+'_nodeLabels.npy', node_labels) #saves the node labels relative to the embeddings files

def classify_embeddings(node2vec_embedding, node_labels, label_map):
    """
    Trains a random forest classifier on node embeddings, using 10 Fold Cross-Validation

    Args:
        node2vec_embedding:numpy array of node embeddings
        node_labels: numpy array of node labels
        label_map: dataframe with the classification label for each userid (columns = ['userid', 'label'])
    Returns:
        model: trained model
        n_scores: 10-fold cross-validation performance
    """
    
    node_labels = node_labels.astype(str)
    
    label_map = users[['userid', 'label']].set_index('userid').T.to_dict('list')
      
    node_colours = []
    
    for target in node_labels:
        if target in label_map.keys():
            node_colours.append(label_map[target][0])
        else: 
            node_colours.append(np.NaN)
    
    df = pd.DataFrame(node2vec_embedding)
    df['label'] = node_colours
    df.dropna(inplace=True)
    
    model = RandomForestClassifier(class_weight='balanced')
    cv = StratifiedKFold(n_splits=10)
    n_scores = cross_validate(model, df[[i for i in range(0, 128)]], df['label'], scoring=['accuracy', 'recall', 'precision','f1', 'roc_auc'], cv=cv, n_jobs=-1, error_score='raise')
    
    return model, n_scores