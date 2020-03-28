#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import networkx as nx
import pandas as pd
df= pd.read_csv('tfr33.csv')
df.dropna(inplace= True)
print(df.shape)


# In[ ]:


df.rename({'combined_score': "weight"}, axis= 1, inplace= True)
G= nx.from_pandas_edgelist(df, 'protein1', 'protein2', ['weight'])
weights = [d['weight'] for s, t, d in G.edges(data=True)]
print(max(weights))
print(G.number_of_edges())
print(G.number_of_nodes())


# In[ ]:


pos = nx.spring_layout(G, weight='weight', k=0.2)


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


def print1():
    elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > 450.0]
    esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] <= 450.0]
    
    import matplotlib.pyplot as plt
    pos = nx.spring_layout(G, weight='weight', k=0.2)
    nx.draw_networkx_edges(G, pos, edgelist=elarge, width=4, alpha=0.5)
    nx.draw_networkx_edges(G, pos, edgelist=esmall, width=4, alpha=0.5, style='dashed')
    nx.draw_networkx_nodes(G, pos, node_color="#bfbf7f", node_shape="h", node_size=6000)
    plt.savefig("part1.png", format="PNG")
    


# In[ ]:





# In[ ]:


def print2():
    weight = [G.edges[e]['weight'] for e in G.edges]

    pos = nx.spring_layout(G, weight='weight', k=0.5)

    nx.draw_networkx_edges(G, pos, edge_color=weight, edge_cmap=plt.cm.Blues, width=4, alpha=0.5)
    nx.draw_networkx_nodes(G, pos, node_color="#bfbf7f", node_shape="h", node_size=3000)

    plt.savefig("part4.png", format="PNG")


# In[ ]:


def calculate():
    closeness = nx.closeness_centrality(G, distance ='weight')
    cl= sorted(closeness.items(), key=lambda x:x[1], reverse=True)[0:10]
    triangles = nx.triangles(G)
    tr= sorted(triangles.items(), key=lambda x:x[1], reverse=True)[0:10]
    eigenvector = nx.eigenvector_centrality(G, max_iter= 5000, weight= "weight")
    eg= sorted(eigenvector.items(), key=lambda x:x[1], reverse=True)[0:10]
    subgraph= nx.subgraph_centrality_exp(G)
    su= sorted(subgraph.items(), key=lambda x:x[1], reverse=True)[0:10]
    tt= pd.DataFrame()
    tt['closeness']= cl
    tt['triangles']= tr
    tt['eigenvector']= eg
    tt['subgraph']= su

    tt.to_csv('network.csv')
    
    
    
    


# In[ ]:


def path_length_histogram():
    # Find path lengths
    length_source_target = dict(nx.shortest_path_length(G))
    # Convert dict of dicts to flat list
    all_shortest = sum([
        list(length_target.values())
        for length_target in length_source_target.values()],
    [])
    # Calculate integer bins
    high = max(all_shortest)
    bins = [-0.5 + i for i in range(high + 2)]
    # Plot histogram
    plt.hist(all_shortest, bins=bins, rwidth=0.8)
    plt.title("path_length_histogram")
    plt.xlabel("Distance")
    plt.ylabel("Count")
    plt.savefig("part5.png", format="PNG")


# In[ ]:


def measure():
    spl= nx.average_shortest_path_length(G, weight= "weight")
    dm= nx.diameter(G)
    tran= nx.transitivity(G)
    avg_cluster= nx.average_clustering(G)
    density= nx.density(G)
    print("graph has sp of {} diameter of {} transitivity of {} avg_clustering of {} and density value {}".format(spl, dm, tran, avg_cluster, density))


# In[ ]:


def cut():
    import networkx.algorithms.connectivity as nxcon
    min_node= nxcon.minimum_node_cut(G)
    min_edge= nxcon.minimum_edge_cut(G)
    print("network minimum cut is node {} and edge{}".format(min_node, min_edge))
    
    


# In[ ]:


# Function to plot a single histogram
def centrality_histogram():
    plt.hist(x, density=True)
    plt.title(title)
    plt.xlabel("Centrality")
    plt.ylabel("Density")


    plt.figure(figsize=(7.5, 2.5))


    centrality_histogram(
    nx.closeness_centrality(G).values(), title="centrality_plot")


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


def gini(x):
    x = [xi for xi in x]
    n = len(x)
    gini_num = sum([sum([abs(x_i - x_j) for x_j in x]) for x_i in x])
    gini_den = 2.0 * n * sum(x)
    return gini_num / gini_den


# In[ ]:


entropy= entropy(nx.closeness_centrality(G, weight= "weight").values())
print("entropy is {}".format(entropy))
gini_new= gini(nx.closeness_centrality(G, weight= "weight").values())
print("gini is {}".format(gini_new))


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


def module():
    from networkx.algorithms.community import greedy_modularity_communities
    communities = sorted(greedy_modularity_communities(G), key=len, reverse=True)

    com= len(communities)
    print(com)
    set_node_community(G, communities)
    set_edge_community(G)
    external = [
    (v, w) for v, w in G.edges
    if G.edges[v, w]['community'] == 0]
    internal = [
    (v, w) for v, w in G.edges
    if G.edges[v, w]['community'] > 0]
    nx.draw_networkx(
    G, pos=pos, node_size=0,
    edgelist=external, edge_color="#333333",
    alpha=0.2, with_labels=False)

    nx.draw_networkx(
    G, pos=pos, node_size=0,
    edgelist=internal, edge_color=internal_color,
    alpha=0.05, with_labels=False)
    plt.savefig("part7.png", format="PNG")

    
    
    


# In[ ]:


def girvan():
    from networkx.algorithms import community
    comp = community.girvan_newman(G)
    communities = next(comp)
    cc= len(communities)
    print(cc)
    set_node_community(G, communities)
    set_edge_community(G)
    node_color = [
    get_color(G.nodes[v]['community'])
    for v in G.nodes]
    external = [
    (v, w) for v, w in G.edges
    if G.edges[v, w]['community'] == 0]
    internal = [
    (v, w) for v, w in G.edges
    if G.edges[v, w]['community'] > 0]
    internal_color = [
    get_color(G.edges[e]['community'])
    for e in internal]
    nx.draw_networkx(
    G, pos=pos, node_size=0,
    edgelist=external, edge_color="#333333", alpha=0.05, with_labels=False)

    nx.draw_networkx(
    G, pos=pos, node_color=node_color,
    edgelist=internal, edge_color=internal_color, alpha=0.05, with_labels=False)
    plt.savefig("part8.png", format="PNG")

    
    


# In[ ]:


def clique():
    cliques = list(nx.find_cliques(G))
    max_clique = max(cliques, key=len)
    print(max_clique)
    node_color = [(0.5, 0.5, 0.5) for v in G.nodes()]
    for i, v in enumerate(G.nodes()):
        if v in max_clique:
            node_color[i] = (0.5, 0.5, 0.9)
    nx.draw_networkx(G, node_color=node_color, pos=pos, alpha=0.05, with_labels=False)
    plt.savefig("part9.png", format="PNG")
    
    
    


# In[ ]:


def k_core():
    G_core_30 = nx.k_core(G, 100)
    G_core_60 = nx.k_core(G, 200)


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


def easy():
    pos = nx.spring_layout(G, k=0.1)
    nx.draw_networkx(
    G, pos=pos, node_size=0,
    edge_color="#333333", alpha=0.05, with_labels=False)
    plt.savefig("part11.png", format="PNG")


# In[ ]:


def markov():
    import markov_clustering as mcl
    import networkx as nx
    adj_matrix = nx.to_numpy_matrix(G)
    res = mcl.run_mcl(adj_matrix)
    clusters = mcl.get_clusters(res)
    mcl.drawing.draw_graph(adj_matrix, clusters, edge_color="red",node_size=40, with_labels=False)
    plt.savefig("part12.png", format="PNG")
    


# In[ ]:


import multiprocessing
from multiprocessing import Process


# In[ ]:


if __name__=="__main__":
    p1=Process(target=print1)
    p2=Process(target=print2)
    p3=Process(target=calculate)
    p4=Process(target=path_length_histogram)
    p5=Process(target=measure)
    p6=Process(target=cut)
    p7=Process(target=centrality_histogram)
    p8=Process(target=tree)
    p9=Process(target=module)
    p10=Process(target=girvan)
    p11=Process(target=clique)
    p12=Process(target=k_core)
    p13=Process(target=easy)
    p14=Process(target=markov)
    
    
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    p6.start()
    p7.start()
    p8.start()
    p9.start()
    p10.start()
    p11.start()
    p12.start()
    p13.start()
    p14.start()
    
    
    
    
    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    p6.join()
    p7.join()
    p8.join()
    p9.join()
    p10.join()
    p11.join()
    p12.join()
    p13.join()
    p14.join()
    
    
    
    print("We're done")


# In[ ]:




