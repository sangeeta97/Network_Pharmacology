#!/usr/bin/env python
# coding: utf-8

# In[2]:


import networkx as nx
import pandas as pd
df= pd.read_csv('tfr33.csv')
df.dropna(inplace= True)

df.rename({'combined_score': "weight"}, axis= 1, inplace= True)
G= nx.from_pandas_edgelist(df, 'protein1', 'protein2', ['weight'])
weights = [d['weight'] for s, t, d in G.edges(data=True)]

pos = nx.spring_layout(G, weight='weight', k=0.2)
import matplotlib.pyplot as plt


# In[3]:


df2 = pd.DataFrame(index=G.nodes())


# In[5]:


betweeness = nx.betweenness_centrality(G, weight ='weight')


# In[6]:


df2['bn']= [i[1] for i in betweeness.items()]


# In[ ]:


df2.to_csv("between.csv")


# In[ ]:


l= list(nx.jaccard_coefficient(G))
l2= list(nx.resource_allocation_index(G))
l3= list(nx.preferential_attachment(G))
common_neigh= [(e[0], e[1], len(list(nx.common_neighbors(G, e[0], e[1])))) for e in nx.non_edges(G)]


# In[ ]:


link= pd.DataFrame()


# In[ ]:


link['jc']= [i[2] for i in l]
link['resource']= [i[2] for i in l2]
link['attachment']= [i[2] for i in l3]
link['common']= [i[2] for i in common_neigh]
link['index']= [i[0:2] for i in l3]
link.set_index("index", inplace= True)
link.to_csv("link.csv")


# In[ ]:


from networkx.algorithms import community
communities = community.greedy_modularity_communities(G)
modularity_dict = {} # Create a blank dictionary
for i,c in enumerate(communities): # Loop through the list of communities, keeping track of the number for the community
    for name in c: # Loop through each person in a community
        modularity_dict[name] = i # Create an entry in the dictionary for the person, where the value is which group they belong to.

# Now you can add modularity information like we did the other metrics
nx.set_node_attributes(G, modularity_dict, 'modularity')


# In[ ]:


betweenness_dict = nx.betweenness_centrality(G, weight ='weight') # Run betweenness centrality
#eigenvector_dict = nx.eigenvector_centrality(G) # Run eigenvector centrality

# Assign each to an attribute in your network
nx.set_node_attributes(G, betweenness_dict, 'betweenness')
#nx.set_node_attributes(G, eigenvector_dict, 'eigenvector')


# In[ ]:


# First get a list of just the nodes in that class
class0 = [n for n in G.nodes() if G.nodes[n]['modularity'] == 0]

# Then create a dictionary of the eigenvector centralities of those nodes
class0_eigenvector = {n:G.nodes[n]['betweenness'] for n in class0}

# Then sort that dictionary and print the first 5 results
class0_sorted_by_eigenvector = sorted(class0_eigenvector.items(), key=itemgetter(1), reverse=True)

print("Modularity Class 0 Sorted by Eigenvector Centrality:")
for node in class0_sorted_by_eigenvector[:5]:
    print("Name:", node[0], "| Eigenvector Centrality:", node[1])


# In[ ]:


for i,c in enumerate(communities): # Loop through the list of communities
    if len(c) > 2: # Filter out modularity classes with 2 or fewer nodes
        print(len(list(c)))


# In[ ]:


nx.write_gexf(G, 'module_network.gexf')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




