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



degree_dict = dict(G.degree(G.nodes()))


# In[ ]:


degree= pd.DataFrame.from_dict(degree_dict)


# In[ ]:


degree.to_csv("degree_nonreal.csv")

