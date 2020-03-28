#!/usr/bin/env python
# coding: utf-8

# In[1]:


import networkx as nx


# In[35]:


import pandas as pd
df= pd.read_csv('tfr.csv')


# In[37]:


df.dropna(inplace= True)


# In[38]:


df.rename({'combined_score': "weight"}, axis= 1, inplace= True)


# In[39]:


G= nx.from_pandas_edgelist(df, 'protein1', 'protein2', ['weight'])


# In[40]:


weights = [d['weight'] for s, t, d in G.edges(data=True)]


# In[41]:


print(max(weights))


# In[42]:


elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > 200.0]
esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] <= 200.0]


# In[43]:


pos = nx.spring_layout(G)


# In[44]:


import matplotlib.pyplot as plt


# In[45]:


# Create figure
plt.figure(figsize=(100,100))
# Calculate layout
pos = nx.spring_layout(G, weight='weight', k=0.9)
# Draw using different shapes and colors for plant/pollinators
nx.draw_networkx_edges(G, pos, edgelist=elarge, width=6, alpha=0.6)

nx.draw_networkx_edges(G, pos, edgelist=esmall,
                       width=3, alpha=0.6, style='dashed')
nx.draw_networkx_nodes(G, pos, node_color="#bfbf7f", node_shape="h", node_size=3000)
plt.savefig("part1.png", format="PNG")


# In[47]:


len(list(nx.connected_components(G)))


# In[ ]:


# Create figure
plt.figure(figsize=(100,100))
# Calculate layout
pos = nx.spring_layout(B, weight='weight', k=0.9)
# Draw using different shapes and colors for plant/pollinators
nx.draw_networkx_edges(B, pos, width=3, alpha=0.2)
nx.draw_networkx_nodes(B, pos, node_color="#bfbf7f", node_shape="h", node_size=3000)
plt.savefig("part2.png", format="PNG")


# In[ ]:


weight = [G.edges[e]['weight'] for e in G.edges]
# Create figure
plt.figure(figsize=(300,300))
# Calculate layout
pos = nx.spring_layout(G, weight='weight', k=0.5)
# Draw edges, nodes, and labels
nx.draw_networkx_edges(G, pos, edge_color=weight, edge_cmap=plt.cm.Blues, width=6, alpha=0.5)
nx.draw_networkx_nodes(G, pos, node_color="#9f9fff", node_size=6000)
nx.draw_networkx_labels(G, pos)
plt.savefig("part3.png", format="PNG")


# In[ ]:


weight = [G.edges[e]['weight'] for e in G.edges]
# Create figure
plt.figure(figsize=(300,300))
# Calculate layout
pos = nx.spring_layout(G, weight='weight', k=0.5)
# Draw edges, nodes, and labels
nx.draw_networkx_edges(G, pos, edge_color=weight, edge_cmap=plt.cm.Blues, width=10, alpha=0.8)
nx.draw_networkx_nodes(G, pos, node_color="#bfbf7f", node_shape="h", node_size=3000)

plt.savefig("part4.png", format="PNG")


# closeness = nx.closeness_centrality(G, distance ='weight')
# cl= sorted(closeness.items(), key=lambda x:x[1], reverse=True)[0:10]

# triangles = nx.triangles(G)
# tr= sorted(triangles.items(), key=lambda x:x[1], reverse=True)[0:10]
#eigenvector = nx.eigenvector_centrality(G, max_iter= 5000, weight= "weight")
#eg= sorted(eigenvector.items(), key=lambda x:x[1], reverse=True)[0:10]subgraph= nx.subgraph_centrality_exp(G)
#su= sorted(subgraph.items(), key=lambda x:x[1], reverse=True)[0:10]
# tt= pd.DataFrame()

# tt['closeness']= cl
# tt['triangles']= tr
# tt['eigenvector']= eg
# tt['subgraph']= su
# 
# tt.to_csv('network.csv')

# In[ ]:





# In[ ]:


def path_length_histogram(G, title=None):
    # Find path lengths
    length_source_target = dict(nx.shortest_path_length(G))
    # Convert dict of dicts to flat list
    all_shortest = sum([
        list(length_target.values())
        for length_target
        in length_source_target.values()],
    [])
    # Calculate integer bins
    high = max(all_shortest)
    bins = [-0.5 + i for i in range(high + 2)]
    # Plot histogram
    plt.hist(all_shortest, bins=bins, rwidth=0.8)
    plt.title(title)
    plt.xlabel("Distance")
    plt.ylabel("Count")
    plt.savefig("part5.png", format="PNG")


# In[ ]:


path_length_histogram(G, title= "path_length_histogram")


# In[ ]:


spl= nx.average_shortest_path_length(G)


# In[ ]:


dm= nx.diameter(G)


# In[ ]:


tran= nx.transitivity(G)


# In[ ]:


avg_cluster= nx.average_clustering(G)


# In[ ]:


density= nx.density(G)


# In[ ]:


import networkx.algorithms.connectivity as nxcon


# In[ ]:


min_node= nxcon.minimum_node_cut(G)


# In[ ]:


min_edge= nxcon.minimum_edge_cut(G)


# In[ ]:


# Function to plot a single histogram
def centrality_histogram(x, title=None):
    plt.hist(x, density=True)
    plt.title(title)
    plt.xlabel("Centrality")
    plt.ylabel("Density")

# Create a figure
plt.figure(figsize=(7.5, 2.5))
# Calculate centralities for each example and plot

centrality_histogram(
    nx.closeness_centrality(G).values(), title="centrality_plot")

# Adjust the layout
plt.tight_layout()
plt.savefig("part6.png", format="PNG")


# In[ ]:


import math
def entropy(x):
    # Normalize
    total = sum(x)
    x = [xi / total for xi in x]
    H = sum([-xi * math.log2(xi) for xi in x])
    return H


# In[ ]:


entropy= entropy(nx.closeness_centrality(G, weight= "weight").values())


# In[ ]:


def gini(x):
    x = [xi for xi in x]
    n = len(x)
    gini_num = sum([sum([abs(x_i - x_j) for x_j in x]) for x_i in x])
    gini_den = 2.0 * n * sum(x)
    return gini_num / gini_den


# In[ ]:


gini_new= gini(nx.closeness_centrality(G, weight= "weight").values())


# In[ ]:


from networkx.algorithms.community import greedy_modularity_communities


# In[ ]:



communities = sorted(greedy_modularity_communities(G), key=len, reverse=True)
# Count the communities
com= len(communities)


# In[ ]:


set_node_community(G, communities)
set_edge_community(G)


# In[ ]:


external = [
    (v, w) for v, w in G.edges
    if G.edges[v, w]['community'] == 0]
internal = [
    (v, w) for v, w in G.edges
    if G.edges[v, w]['community'] > 0]


# In[ ]:


def get_color(i, r_off=1, g_off=1, b_off=1):
    r0, g0, b0 = 0, 0, 0
    n = 16
    low, high = 0.1, 0.9
    span = high - low
    r = low + span * (((i + r_off) * 3) % n) / (n - 1)
    g = low + span * (((i + g_off) * 5) % n) / (n - 1)
    b = low + span * (((i + b_off) * 7) % n) / (n - 1)
    return (r, g, b)


# In[ ]:


internal_color = [
    get_color(G.edges[e]['community'])
    for e in internal]


# In[ ]:


# Draw external edges
nx.draw_networkx(
    G, pos=pos, node_size=0,
    edgelist=external, edge_color="#333333",
    alpha=0.2, with_labels=False)
# Draw internal edges
nx.draw_networkx(
    G, pos=pos, node_size=0,
    edgelist=internal, edge_color=internal_color,
    alpha=0.05, with_labels=False)
plt.savefig("part7.png", format="PNG")


# In[ ]:


from networkx.algorithms import community


# In[ ]:


comp = community.girvan_newman(G)


# In[ ]:


communities = next(comp)
cc= len(communities)


# In[ ]:


set_node_community(G, communities)
set_edge_community(G)
node_color = [
    get_color(G.nodes[v]['community'])
    for v in G.nodes]


# In[ ]:


# Set community color for internal edges
external = [
    (v, w) for v, w in G.edges
    if G.edges[v, w]['community'] == 0]
internal = [
    (v, w) for v, w in G.edges
    if G.edges[v, w]['community'] > 0]
internal_color = [
    get_color(G.edges[e]['community'])
    for e in internal]


# In[ ]:


# Draw external edges
nx.draw_networkx(
    G, pos=pos, node_size=0,
    edgelist=external, edge_color="#333333", alpha=0.05, with_labels=False)
# Draw nodes and internal edges
nx.draw_networkx(
    G, pos=pos, node_color=node_color,
    edgelist=internal, edge_color=internal_color, alpha=0.05, with_labels=False)
plt.savefig("part8.png", format="PNG")


# In[ ]:


cliques = list(nx.find_cliques(G))


# In[ ]:


max_clique = max(cliques, key=len)


# In[ ]:


# Visualize maximum clique
node_color = [(0.5, 0.5, 0.5) for v in G.nodes()]
for i, v in enumerate(G.nodes()):
    if v in max_clique:
        node_color[i] = (0.5, 0.5, 0.9)
nx.draw_networkx(G, node_color=node_color, pos=pos, alpha=0.05, with_labels=False)
plt.savefig("part9.png", format="PNG")


# In[ ]:


# Find k-cores
G_core_30 = nx.k_core(G, 100)
G_core_60 = nx.k_core(G, 200)

# Visualize network and k-cores
nx.draw_networkx(
    G, pos=pos, node_size=0,
    edge_color="#333333", alpha=0.05, with_labels=False)
nx.draw_networkx(
    G_core_30, pos=pos, node_size=0,
    edge_color="#7F7FEF", alpha=0.05, with_labels=False)
nx.draw_networkx(
    G_core_60, pos=pos, node_size=0,
    edge_color="#AFAF33", alpha=0.05, with_labels=False)
plt.savefig("part10.png", format="PNG")


# In[ ]:


av_path= nx.average_shortest_path_length(G, weight= "weight")


# In[ ]:


# Calculate layout and draw
pos = nx.spring_layout(G, k=0.1)
nx.draw_networkx(
    G, pos=pos, node_size=0,
    edge_color="#333333", alpha=0.05, with_labels=False)
plt.savefig("part11.png", format="PNG")


# In[ ]:


import markov_clustering as mcl
import networkx as nx

adj_matrix = nx.to_numpy_matrix(G)

res = mcl.run_mcl(adj_matrix)
clusters = mcl.get_clusters(res)

mcl.drawing.draw_graph(adj_matrix, clusters, edge_color="red",node_size=40, with_labels=False)
plt.savefig("part12.png", format="PNG")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




