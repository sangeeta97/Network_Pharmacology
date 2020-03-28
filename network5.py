#!/usr/bin/env python
# coding: utf-8

# In[1]:


import networkx as nx
import pandas as pd
df= pd.read_csv('tfr33.csv')
df.dropna(inplace= True)

df.rename({'combined_score': "weight"}, axis= 1, inplace= True)
G= nx.from_pandas_edgelist(df, 'protein1', 'protein2', ['weight'])
weights = [d['weight'] for s, t, d in G.edges(data=True)]

pos = nx.spring_layout(G, weight='weight', k=0.2)
import matplotlib.pyplot as plt


# In[5]:


degree_dict = dict(G.degree(G.nodes()))


# In[10]:


new= pd.DataFrame()


# In[11]:


new['protein']= degree_dict.keys()


# In[12]:


new['degree']= degree_dict.values()


# In[ ]:


degree.to_csv("degree.csv")


# In[19]:


from operator import itemgetter


# In[18]:


sorted_degree = sorted(degree_dict.items(), key=itemgetter(1), reverse=True)
print("Top 20 nodes by degree:")
for d in sorted_degree[:20]:
    print(d)


# In[17]:


import csv
from operator import itemgetter


# In[35]:


from community import community_louvain
partition = community_louvain.best_partition(G)


# In[38]:


size = float(len(set(partition.values())))
pos = nx.spring_layout(G)
count = 0
for com in set(partition.values()) :
    count = count + 1
    list_nodes = [nodes for nodes in partition.keys()
                                if partition[nodes] == com]
    nx.draw_networkx_nodes(G, pos, list_nodes, node_size = 20,
                                node_color = str(count / size))


nx.draw_networkx_edges(G, pos, alpha=0.5)
plt.savefig("partition.png", format="PNG")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




