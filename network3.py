#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import networkx as nx
import pandas as pd
df= pd.read_csv('tfr33.csv')
df.dropna(inplace= True)

df.rename({'combined_score': "weight"}, axis= 1, inplace= True)
G= nx.from_pandas_edgelist(df, 'protein1', 'protein2', ['weight'])
weights = [d['weight'] for s, t, d in G.edges(data=True)]

pos = nx.spring_layout(G, weight='weight', k=0.2)
import matplotlib.pyplot as plt


# In[ ]:


import csv
from operator import itemgetter


# In[ ]:


def node():
    G.nodes(data=True)
    df = pd.DataFrame(index=G.nodes())
    df['clustering'] = pd.Series(nx.clustering(G))
    df['degree'] = pd.Series(G.degree())
    df.to_csv("node.csv")
    
    
    


# In[ ]:


def edge():
    G.edges(data=True)
    df1 = pd.DataFrame(index=G.edges())
    df1['weight'] = pd.Series(nx.get_edge_attributes(G, 'weight'))
    df1['preferential attachment'] = [i[2] for i in nx.preferential_attachment(G, df1.index)]
    df1['Common Neighbors'] = df1.index.map(lambda city: len(list(nx.common_neighbors(G, city[0], city[1]))))
    df1['preds_jaccard'] = [i[2] for i in nx.jaccard_coefficient(G, df1.index)]
    df1['preds_resource_allocation'] = [i[2] for i in nx.resource_allocation_index(G df1.index)]
    df1.to_csv("edge.csv")
    


# In[ ]:


def calculate():
    df2 = pd.DataFrame(index=G.nodes())
    closeness = nx.closeness_centrality(G, distance ='weight')
    df2['cl']= [i[1] for i in closeness.items()]
    triangles = nx.triangles(G)
    df2['tr'] = [i[1] for i in triangles.items()]
    eigenvector = nx.eigenvector_centrality(G, max_iter= 5000, weight= "weight")
    df2['eg']= [i[1] for i in eigenvector.items()]
    subgraph= nx.subgraph_centrality_exp(G)
    df2['su'] = [i[1] for i in subgraph.items()]
    

    df2.to_csv('network_full.csv')
    
    
    


# In[ ]:


def module():
    from networkx.algorithms.community import greedy_modularity_communities
    communities = community.greedy_modularity_communities(G)
    
    eigenvector_dict = nx.eigenvector_centrality(G, max_iter= 5000, weight= "weight")
    
    nx.set_node_attributes(G, eigenvector_dict, 'eigenvector')
    modularity_dict = {}
    for i,c in enumerate(communities):
        for name in c:
            modularity_dict[name] = i
    nx.set_node_attributes(G, modularity_dict, 'modularity')
    class0 = [n for n in G.nodes() if G.nodes[n]['modularity'] == 0]
    class0_eigenvector = {n:G.nodes[n]['eigenvector'] for n in class0}
    class0_sorted_by_eigenvector = sorted(class0_eigenvector.items(), key=itemgetter(1), reverse=True)
    print("Modularity Class 0 Sorted by Eigenvector Centrality:")
    for node in class0_sorted_by_eigenvector[:5]:
        print("Name:", node[0], "| Eigenvector Centrality:", node[1])
    for i,c in enumerate(communities):
        if len(c) > 2:
            print('Class '+str(i)+':', list(c))
            
        
    
    
    
            
    
    
    


# In[ ]:


if __name__=="__main__":
    p1=Process(target=node)
    p2=Process(target=edge)
    p3=Process(target=calculate)
    p4=Process(target=module)
    
    
    
    p1.start()
    p2.start()
    p3.start()
    p4.start()
       
    
    
    p1.join()
    p2.join()
    p3.join()
    p4.join()
    
    
    
    print("We're done")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# from networkx.algorithms.community import greedy_modularity_communities
# communities = community.greedy_modularity_communities(G)
# 

# betweenness_dict = nx.betweenness_centrality(G) # Run betweenness centrality
# eigenvector_dict = nx.eigenvector_centrality(G, max_iter= 5000, weight= "weight") # Run eigenvector centrality
# 
# # Assign each to an attribute in your network
# nx.set_node_attributes(G, betweenness_dict, 'betweenness')
# nx.set_node_attributes(G, eigenvector_dict, 'eigenvector')
# 

# modularity_dict = {} # Create a blank dictionary
# for i,c in enumerate(communities): # Loop through the list of communities, keeping track of the number for the community
#     for name in c: # Loop through each person in a community
#         modularity_dict[name] = i # Create an entry in the dictionary for the person, where the value is which group they belong to.
# 
# # Now you can add modularity information like we did the other metrics
# nx.set_node_attributes(G, modularity_dict, 'modularity')
# 

# # First get a list of just the nodes in that class
# class0 = [n for n in G.nodes() if G.nodes[n]['modularity'] == 0]
# 
# # Then create a dictionary of the eigenvector centralities of those nodes
# class0_eigenvector = {n:G.nodes[n]['eigenvector'] for n in class0}
# 
# # Then sort that dictionary and print the first 5 results
# class0_sorted_by_eigenvector = sorted(class0_eigenvector.items(), key=itemgetter(1), reverse=True)
# 
# print("Modularity Class 0 Sorted by Eigenvector Centrality:")
# for node in class0_sorted_by_eigenvector[:5]:
#     print("Name:", node[0], "| Eigenvector Centrality:", node[1])
# 

# for i,c in enumerate(communities): # Loop through the list of communities
#     if len(c) > 2: # Filter out modularity classes with 2 or fewer nodes
#         print('Class '+str(i)+':', list(c)) # Print out the classes and their members
# 

# In[ ]:





# In[ ]:





# In[ ]:




