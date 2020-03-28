#!/usr/bin/env python
# coding: utf-8

#!/usr/bin/env python
# coding: utf-8
'''import libraries for mcl clustering and plotting'''

get_ipython().run_line_magic('matplotlib', 'notebook')
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

import markov_clustering as mc
import networkx as nx
import os
os.getcwd()

'''import edge table in pandas dataframe'''

import pandas as pd
df= pd.read_csv('/home/ibab/Desktop/present/curated_network+mcl/imp_node_edge_parent_network.csv')
df.dropna(inplace= True)

'''Plotting the network for with mapping of different node attributes'''

G= nx.from_pandas_edgelist(df, 'source', 'target', ['weight'])
weights = [d['weight'] for s, t, d in G.edges(data=True)]
pos = nx.spring_layout(G, weight='weight', k=0.2)
import matplotlib.pyplot as plt
import math
get_ipython().run_line_magic('matplotlib', 'notebook')
edge_width= [.00001*G[u][v]["weight"] for u,v in G.edges()]
node_size= [(4000*nx.get_node_attributes(G, "closeness")[v])**2 for v in G]
node_color= [math.sqrt(i[1]) for i in degree_dict.items()]

'''Plotting network with continous mapping degree on node size, weight of edge width and closeness centrality as node color''' 
nx.draw_networkx(
    G, 
     with_labels=False, node_size=node_color, alpha= 0.7, node_color= node_size, width= edge_width, cmap= plt.cm.Blues)
plt.savefig("imp_4000_degree+weight+closeness.png")
plt.show()

'''Running MCL algorithm to find modules of network after optimizing the inflation parameter'''

for inflation in [i / 10 for i in range(15, 26)]:
     result = mc.run_mcl(A, inflation=inflation)
     clusters = mc.get_clusters(result)
     result= np.asmatrix(result)
     Q = mc.modularity(matrix=result, clusters=clusters)
     print("inflation:", inflation, "modularity:", Q)


# Run MCL algorithm
A = nx.to_numpy_matrix(G)
result = mc.run_mcl(A, inflation=1.7)
clusters = mc.get_clusters(result)

len(clusters)

