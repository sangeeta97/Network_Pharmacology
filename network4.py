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


import community


# In[ ]:


degree_dict = dict(G.degree(G.nodes()))


# In[ ]:


degree= pd.DataFrame.from_dict(degree_dict)


# In[ ]:


degree.to_csv("degree.csv")


# In[ ]:


sorted_degree = sorted(degree_dict.items(), key=itemgetter(1), reverse=True)
print("Top 20 nodes by degree:")
for d in sorted_degree[:20]:
    print(d)


# In[ ]:


import csv
from operator import itemgetter


# In[ ]:


import multiprocessing
from multiprocessing import Process


# In[ ]:


def plot():
    import community
    import networkx as nx
    import matplotlib.pyplot as plt

#better with karate_graph() as defined in networkx example.
#erdos renyi don't have true community structure
    G = nx.erdos_renyi_graph(30, 0.05)

#first compute the best partition
    partition = community.best_partition(G)

#drawing
    size = float(len(set(partition.values())))
    pos = nx.spring_layout(G)
    count = 0.
    for com in set(partition.values()) :
        count = count + 1.
        list_nodes = [nodes for nodes in partition.keys()
                                if partition[nodes] == com]
        nx.draw_networkx_nodes(G, pos, list_nodes, node_size = 20,
                                node_color = str(count / size))


    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.show()


# In[ ]:


if __name__=="__main__":
    p1=Process(target=plot)
    
    
    p1.start()
   
    
    p1.join()
    
    
    print("We're done")


# In[ ]:





# In[ ]:




