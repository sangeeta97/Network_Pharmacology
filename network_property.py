#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import networkx as nx
import pandas as pd
df= pd.read_csv('tfr.csv')
df.dropna(inplace= True)

'''Import csv edge table in NetworkX network'''

df.rename({'combined_score': "weight"}, axis= 1, inplace= True)
G= nx.from_pandas_edgelist(df, 'protein1', 'protein2', ['weight'])
weights = [d['weight'] for s, t, d in G.edges(data=True)]

pos = nx.spring_layout(G, weight='weight', k=0.2)
import matplotlib.pyplot as plt



import csv
from operator import itemgetter

'''Creating a Function to calculate all the edge property for the true edge present in the network by passing index (edge) as input'''

def edge():
    G.edges(data=True)
    df1 = pd.DataFrame(index=G.edges())
    df1['weight'] = pd.Series(nx.get_edge_attributes(G, 'weight'))
    df1['preferential attachment'] = [i[2] for i in nx.preferential_attachment(G, df1.index)]
    df1['Common Neighbors'] = df1.index.map(lambda city: len(list(nx.common_neighbors(G, city[0], city[1]))))
    df1['preds_jaccard'] = [i[2] for i in nx.jaccard_coefficient(G, df1.index)]
    df1['preds_resource_allocation'] = [i[2] for i in nx.resource_allocation_index(G df1.index)]
    df1.to_csv("edge.csv")
    
''' Calculating node property and Centrality measures of true network by creating a function'''


def calculate():
    df2 = pd.DataFrame(index=G.nodes())
    closeness = nx.closeness_centrality(G, distance ='weight')
    df2['cl']= [i[1] for i in closeness.items()]
    triangles = nx.triangles(G)
    df2['tr'] = [i[1] for i in triangles.items()]
    eigenvector = nx.eigenvector_centrality(G, max_iter= 5000, weight= "weight")
    df2['eg']= [i[1] for i in eigenvector.items()]    
    betweeness = nx.betweenness_centrality(G, weight ='weight')
    nx.set_node_attributes(G, betweeness, 'betweeness')
    df2['bn']= [i[1] for i in betweeness.items()]
    clustering = pd.Series(nx.clustering(G, weight= "weight"))
    nx.set_node_attributes(G, clustering, 'clustering')
    df2['clustering']= [i[1] for i in clustering.items()]
    df2['degree'] = pd.Series(G.degree())
    eccentricity = pd.Series(nx.eccentricity(G))
    nx.set_node_attributes(G, eccentricity, 'eccentricity')
    df2['eccentricity']= [i[1] for i in eccentricity.items()]
    df2.to_csv('network_node.csv')
    
    
''' Importing multiprocessing module to run both functions on different processor for faster excution'''
    
import multiprocessing
from multiprocessing import Process 
if __name__=="__main__":
    p1=Process(target=calculate)
    p2=Process(target=edge)
       
    
    p1.start()
    p2.start()
        
       
    p1.join()
    p2.join()
     
       
    print("We're done")

 
