#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


import networkx as nx
import pandas as pd
df= pd.read_csv('link_median_attach.csv')
df.dropna(inplace= True)


# In[3]:


df['protein1']= df['index'].map(lambda x: str(x)).str.replace("\W+", " ").str.split().map(lambda x: x[0])


# In[4]:


df['protein2']= df['index'].map(lambda x: str(x)).str.replace("\W+", " ").str.split().map(lambda x: x[-1])


# In[5]:


df= df[['protein1', 'protein2', 'attachment']]


# In[13]:


df.rename({'attachment': "weight"}, axis= 1, inplace= True)
G= nx.from_pandas_edgelist(df, 'protein1', 'protein2', ['weight'])
weights = [d['weight'] for s, t, d in G.edges(data=True)]

pos = nx.spring_layout(G, weight='weight', k=0.2)
import matplotlib.pyplot as plt


# In[ ]:


def calculate():
    global G
    df2 = pd.DataFrame(index=G.nodes())
    closeness = nx.closeness_centrality(G, distance ='weight')
    nx.set_node_attributes(G, closeness, 'closeness')
    df2['cl']= [i[1] for i in closeness.items()]
    triangles = nx.triangles(G)
    nx.set_node_attributes(G, triangles, 'triangles')
    df2['tr'] = [i[1] for i in triangles.items()]
    eigenvector = nx.eigenvector_centrality(G, max_iter= 5000, weight= "weight")
    nx.set_node_attributes(G, eigenvector, 'eigenvector')
    df2['eg']= [i[1] for i in eigenvector.items()]
    subgraph= nx.subgraph_centrality_exp(G)
    nx.set_node_attributes(G, subgraph, 'subgraph')
    df2['su'] = [i[1] for i in subgraph.items()]
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

    df2.to_csv('network_nonreal.csv')
    


# In[ ]:


def markov():
    global G
    import markov_clustering as mcl
    import networkx as nx
    adj_matrix = nx.to_numpy_matrix(G)
    res = mcl.run_mcl(adj_matrix)
    clusters = mcl.get_clusters(res)
    cluster_dict = {}
    for i,c in enumerate(clusters):
        for name in c:
            cluster_dict[name] = i
    nx.set_node_attributes(G, cluster_dict, 'cluster')
            
        
    
    
    
    
    


# In[ ]:


def module():
    global G
    from networkx.algorithms import community
    communities = community.greedy_modularity_communities(G)
    modularity_dict = {}
    for i,c in enumerate(communities):
        for name in c:
            modularity_dict[name] = i
            
    nx.set_node_attributes(G, modularity_dict, 'modularity')
            
        
    
    
    
    


# In[ ]:


def partition():
    global G
    from networkx.algorithms import community
    from community import community_louvain
    partitions = community_louvain.best_partition(G)
    partition_dict = {}
    for i,c in enumerate(partitions):
        for name in c:
            partition_dict[name] = i
    nx.set_node_attributes(G, partition_dict, 'partition')
            
        
    
    
    
    


# In[ ]:


df1= pd.read_csv('tfr33.csv')
df1.dropna(inplace= True)

df1.rename({'combined_score': "weight"}, axis= 1, inplace= True)
G1= nx.from_pandas_edgelist(df1, 'protein1', 'protein2', ['weight'])
weights = [d['weight'] for s, t, d in G1.edges(data=True)]



# In[ ]:


def calculate1():
    global G1
    df2 = pd.DataFrame(index=G1.nodes())
    closeness = nx.closeness_centrality(G1, distance ='weight')
    nx.set_node_attributes(G1, closeness, 'closeness')
    df2['cl']= [i[1] for i in closeness.items()]
    triangles = nx.triangles(G1)
    nx.set_node_attributes(G1, triangles, 'triangles')
    df2['tr'] = [i[1] for i in triangles.items()]
    eigenvector = nx.eigenvector_centrality(G1, max_iter= 5000, weight= "weight")
    nx.set_node_attributes(G1, eigenvector, 'eigenvector')
    df2['eg']= [i[1] for i in eigenvector.items()]
    subgraph= nx.subgraph_centrality_exp(G1)
    nx.set_node_attributes(G1, subgraph, 'subgraph')
    df2['su'] = [i[1] for i in subgraph.items()]
    betweeness = nx.betweenness_centrality(G1, weight ='weight')
    nx.set_node_attributes(G1, betweeness, 'betweeness')
    df2['bn']= [i[1] for i in betweeness.items()]
    clustering = pd.Series(nx.clustering(G1, weight= "weight"))
    nx.set_node_attributes(G1, clustering, 'clustering')
    df2['clustering']= [i[1] for i in clustering.items()]
    df2['degree'] = pd.Series(G1.degree())
    eccentricity = pd.Series(nx.eccentricity(G1))
    nx.set_node_attributes(G1, eccentricity, 'eccentricity')
    df2['eccentricity']= [i[1] for i in eccentricity.items()]

    df2.to_csv('network_real.csv')
    


# In[ ]:


def markov1():
    global G1
    import markov_clustering as mcl
    import networkx as nx
    adj_matrix = nx.to_numpy_matrix(G1)
    res = mcl.run_mcl(adj_matrix)
    clusters = mcl.get_clusters(res)
    cluster_dict = {}
    for i,c in enumerate(clusters):
        for name in c:
            cluster_dict[name] = i
    nx.set_node_attributes(G1, cluster_dict, 'cluster')
            
        


# In[ ]:


def module1():
    global G1
    from networkx.algorithms import community
    communities = community.greedy_modularity_communities(G1)
    modularity_dict = {}
    for i,c in enumerate(communities):
        for name in c:
            modularity_dict[name] = i
            
    nx.set_node_attributes(G1, modularity_dict, 'modularity')
            


# In[ ]:


def partition1():
    global G1
    from networkx.algorithms import community
    from community import community_louvain
    partitions = community_louvain.best_partition(G1)
    partition_dict = {}
    for i,c in enumerate(partitions):
        for name in c:
            partition_dict[name] = i
    nx.set_node_attributes(G1, partition_dict, 'partition')
            


# In[ ]:


import multiprocessing
from multiprocessing import Process


# In[ ]:


if __name__=="__main__":
    p1=Process(target=calculate)
    p2=Process(target=markov)
    p3=Process(target=module)
    p4=Process(target=partition)
    p5=Process(target=calculate1)
    p6=Process(target=markov1)
    p7=Process(target=module1)
    p8=Process(target=partition1)
    
    
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    p6.start()
    p7.start()
    p8.start()

    
    
    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    p6.join()
    p7.join()
    p8.join()
    
    
    
    print("We're done")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




