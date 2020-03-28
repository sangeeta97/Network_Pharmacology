#!/usr/bin/env python
# coding: utf-8

''' Reading the csv file in Pandas dataFrame '''

import networkx as nx
import pandas as pd
df= pd.read_csv('tfr.csv')
df.dropna(inplace= True)
df.rename({'combined_score': "weight"}, axis= 1, inplace= True)

'''Reading the Pandas DataFrame in NetworkX for network'''

G= nx.from_pandas_edgelist(df, 'protein1', 'protein2', ['weight'])
weights = [d['weight'] for s, t, d in G.edges(data=True)]
print(max(weights))
pos = nx.spring_layout(G)


'''Calculating the Network property using NetworkX topological measures'''
spl= nx.average_shortest_path_length(G)
dm= nx.diameter(G)
tran= nx.transitivity(G)
avg_cluster= nx.average_clustering(G)
density= nx.density(G)

import matplotlib.pyplot as plt

'''Plotting the shrotest path length of all the nodes as histogram'''

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

path_length_histogram(G, title= "path_length_histogram")



