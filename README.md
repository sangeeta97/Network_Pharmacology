# Network_Pharmacology
Network Data Analysis using NetworkX. All the script files to reproduce this study is provided.


EDA_unsupervised.py contains the code used to create plots listed in All_data/node_plots and All_data/edge_score_plots folders. 

mcl_clustering.py contains code to find module in a network.

network_Topology.py contains code to meaure whole netwotk atrributes like average clustering, diameter, shortest path length, transitivity and density. 

network_property.py contains code to measure node scores such as betweenness centrality, closeness centrality, degree, transitivity, eecentricity, and eignvector centrality. Also it contains codes for calculating edge score by using link prediction algorithms, such as jaccard index, preferential attachment score, common neighbor score and resource allocation score. 

# All_data

This study’s primary goal is unraveling the mechanism of action of bioactives of Curcuma longa L. at the molecular level using protein-protein interaction network.
The target proteins (TP) were obtained using similarity ensemble approach and they were further quried in StringDB for retriving the intraction proteins (IP) and a network graph called as true PPIN was created using Networkx library.
Another Network was created by using all the non-existent edges between the list of TP and IP called as false PPIN. 
The PPIN topological measure as edge scores and node scores were calculated and compared between true PPIN and false PPIN. The exploratory data analysis was performed. We identified closeness centrality as important node attribute and jaccard index as important edge attribute for a true PPIN. 

Our All_data folder contains data for all the steps  involved in this project work. The step-wise analysis is explained below with the name of the respective folders containing the data.  

__Step 1: Bioactives_target_proteins__ This folder contains 3 csv files namely Bisdemethoxycurcumin_target_proteins, curcumin_target_proteins and desmethoxycurcumin_target_proteins obtained by searching respective bioactive compounds in (http://sea.bkslab.org/) which gives a list of putative target proteins using similarity ensemble approach. 

__Step 2: Target_proteins+Interacting_proteins__

* A combined list of target proteins (219) were queried in the StringDB protein-protein interaction database for human.
  
* This has led to 208125 interactions for which the interaction score was varied from 150 to 999. Further to reduce the complexity of the network and to increase the confidence of the interaction, we included only edges having interaction score above 300. This has led to total 58482 interactions (edge) involving 11979 proteins (nodes). Out of 11979 proteins, 219 were target proteins (TP) and rest were interacting proteins (TP). These interaction were tabluated in edge.csv in which first column contains TP and second column contains IP. These 58482 interactions were listed in the edge_list_true_PPIN.csv. Also, another network  was created using non-existent interaction as edges and shown in the edge_list_false_PPIN.csv. 

__Step 3: True_PPIN_edge_score & True_PPIN_node_score__

* The node scores values such as betweenness centrality, closeness centrality, degree, transitivity, eecentricity, and eignvector centrality are presented along with the their node name in True_PPIN_node_score. 
  
* The edge score calculated by using link prediction algorithms, such as jaccard index, preferential attachment score, common neighbor score and resource allocation score are listed in True_PPIN_edge_score. 

__step 4: False_PPIN_edge_score & False_PPIN_node_score__
* The node scores values such as betweenness centrality, closeness centrality, degree, transitivity, eecentricity, and eignvector centrality are presented along with the their node name in False_PPIN_node_score. 
  
* The edge score calculated by using link prediction algorithms, such as jaccard index, preferential attachment score, common neighbor score and resource allocation score are listed in True_PPIN_edge_score. 

__Step 5: Edge_score_plots & Node_plots__ A exploratory analysis of comparative score values (Node and edge attributes) for True_PPIN and False_PPIN using boxplot, lmplots, facegrid, and heatmaps. 

__Step 6: mcl_clustering_input_data__ we removed the insignificant edges and nodes from True_PPIN and made it sparse. We removed edges having jaccard score value above 75 percentile of True_PPIN and nodes that had closeness centrality value less than the 25 percentiles. The resulting network had 1900 nodes and 4637 edges. This data is presented in a csv file. This edgelist was passed as input to create a sparse network and further mcl_clustering algorithm was applied to this network to detect the protein modules and their corresponding pathways. 





time.

