#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re
from pandas import plotting

# for visualizations
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')


# In[20]:


df1= pd.read_csv("edge.csv")


# In[12]:


df2= pd.read_csv("link_median_attach.csv")


# In[26]:


df2.head()


# In[22]:



df1= df1.drop(['Unnamed: 0', 'Unnamed: 1'], axis= 1)


# In[23]:


df1.drop(['weight'], axis= 1, inplace= True)


# In[25]:


df1.columns= ['attachment', 'common', 'jc', 'resource']


# In[27]:


df2.drop(["Unnamed: 0", "index"],axis= 1, inplace= True)


# In[32]:


df1.head()


# In[29]:


dfall= pd.concat([df1, df2], keys= ['true_edge', 'false_edge'])
dfall.reset_index(inplace= True)


# In[35]:


dfall.drop(["level_1"], axis= 1, inplace= True)


# In[33]:


dfall.rename({"level_0": "edge_type"}, axis= 1, inplace= True)


# In[37]:


df1= dfall


# In[ ]:


df1.hist(edgecolor='black', linewidth=1.2)
fig=plt.gcf()
fig.set_size_inches(15,6)

plt.savefig("hist9.png")


# In[ ]:


plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.boxplot(x='edge_type',y='attachment',data=df1)
plt.subplot(2,2,2)
sns.boxplot(x='edge_type',y='common',data=df1)
plt.subplot(2,2,3)
sns.boxplot(x='edge_type',y='jc',data=df1)
plt.subplot(2,2,4)
sns.boxplot(x='edge_type',y='resource',data=df1)
plt.savefig("box9.png")


# In[ ]:


sns.lmplot(x="jc", y="attachment",hue="edge_type",data=df1)
plt.savefig("lmplot91.png")


# In[ ]:


sns.lmplot(x="common", y="jc",hue="edge_type",data=df1)
plt.savefig("lmplot92.png")


# In[ ]:


sns.lmplot(x="common", y="resource",hue="edge_type",data=df1)
plt.savefig("lmplot93.png")


# In[ ]:


sns.pairplot(data=df1,hue="edge_type",palette="Set1")
plt.suptitle("Pair Plot of edge type",fontsize=20)
plt.savefig("pairplot9.png")


# In[ ]:


ff= df1.iloc[:, 1:5]


# In[ ]:


df1.edge_type.value_counts()


# In[ ]:


from sklearn import preprocessing

scaler = preprocessing.StandardScaler()

scaler.fit(ff)
X_scaled_array = scaler.transform(ff)
X_scaled = pd.DataFrame(X_scaled_array, columns = ff.columns)


# In[ ]:


from sklearn.decomposition import PCA

ndimensions = 2

pca = PCA(n_components=ndimensions, random_state=50)
pca.fit(X_scaled)
X_pca_array = pca.transform(X_scaled)
X_pca = pd.DataFrame(X_pca_array, columns=['PC1','PC2']) 


# In[ ]:


yy= df1.iloc[:, 0]


# In[ ]:


yy= yy.map({"true_edge": 0, "false_edge": 1})


# In[ ]:


from sklearn.cluster import KMeans

nclusters = 2 # this is the k in kmeans
seed = 0

km = KMeans(n_clusters=nclusters, random_state=seed)
km.fit(X_scaled)

# predict the cluster for each data point
y_cluster_kmeans = km.predict(X_scaled)


# In[5]:


import time


# In[ ]:


y_id_array = np.array(yy)
df_plot = X_pca.copy()
df_plot['ClusterKmeans'] = y_cluster_kmeans
df_plot['edge_type'] = y_id_array # also add actual labels so we can use it in later plots
df_plot.to_csv("df1_plot.csv")


# In[ ]:


import matplotlib as mpl


# In[ ]:


def plotData(df, groupby):
    "make a scatterplot of the first two principal components of the data, colored by the groupby field"
    
    # make a figure with just one subplot.
    # you can specify multiple subplots in a figure, 
    # in which case ax would be an array of axes,
    # but in this case it'll just be a single axis object.
    fig, ax = plt.subplots(figsize = (7,7))

    # color map
    cmap = mpl.cm.get_cmap('prism')

    # we can use pandas to plot each cluster on the same graph.
    # see http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.plot.html
    for i, cluster in df.groupby(groupby):
        cluster.plot(ax = ax, # need to pass this so all scatterplots are on same graph
                     kind = 'scatter', 
                     x = 'PC1', y = 'PC2',
                     color = cmap(i/(nclusters-1)), # cmap maps a number to a color
                     label = "%s %i" % (groupby, i), 
                     s=30) # dot size
    ax.grid()
    ax.axhline(0, color='black')
    ax.axvline(0, color='black')
    ax.set_title("Principal Components Analysis (PCA) of edge types")
    plt.savefig("pca" + str(round(time.time())));

# plot the clusters each datapoint was assigned to
plotData(df_plot, 'ClusterKmeans')


# In[ ]:


plotData(df_plot, 'edge_type')


# In[ ]:


from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=nclusters)
gmm.fit(X_scaled)

# predict the cluster for each data point
y_cluster_gmm = gmm.predict(X_scaled)
y_cluster_gmm


# In[ ]:


df_plot['ClusterGMM'] = y_cluster_gmm
plotData(df_plot, 'ClusterGMM')


# In[ ]:


pca = PCA().fit(X_scaled)#Plotting the Cumulative Summation of the Explained Variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Explained Variance')
plt.savefig("pca_variance")


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.metrics import roc_auc_score,  roc_curve, auc



from IPython.display import display


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(ff, yy, 
                                                       test_size = 0.25, stratify = yy,
                                                       shuffle = True)


       # Fitting Random Forest Classification to the Training set
rfc = RandomForestClassifier(n_estimators=20,
                                max_depth=None,
                                min_samples_split=15,
                                min_samples_leaf=5,
                                max_features=0.5,
                                n_jobs=-1,
                                random_state=42)
rfc.fit(X_train, y_train)


# In[ ]:


acc_rf = []
auc_rf = []


# In[ ]:


feature_importance_rf = pd.DataFrame()
y_pred = rfc.predict(X_test)
    
    #print(pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Prediction']))
    
acc_rf.append(accuracy_score(y_test, y_pred))
    
y_pred_prob = rfc.predict_proba(X_test)
fpr, tpr, t = roc_curve(y_test, y_pred_prob[:,1])
auc_s = auc(fpr, tpr)
auc_rf.append(auc_s)
    
fold_importance_df = pd.DataFrame()
fold_importance_df["feature"] = ff.columns
fold_importance_df["importance"] = rfc.feature_importances_

feature_importance_rf = pd.concat([feature_importance_rf, fold_importance_df], axis=0)


# In[ ]:


print(f"---------- Accuracy ------------ \n      Max: {np.max(acc_rf)} \n      Min: {np.min(acc_rf)} \n      Mean: {np.mean(acc_rf)}\n      Std: {np.std(acc_rf)}")

print(f"---------- AUC Score ------------ \n      Max: {np.max(auc_rf)} \n      Min: {np.min(auc_rf)} \n      Mean: {np.mean(auc_rf)}\n      Std: {np.std(auc_rf)}")


# In[ ]:


sns.set_style('whitegrid')

cols = (feature_importance_rf[["feature", "importance"]]
    .groupby("feature")
    .mean()
    .sort_values(by="importance", ascending=False)[:30].index)

best_features = feature_importance_rf.loc[feature_importance_rf['feature'].isin(cols)]

plt.figure(figsize=(10,10))
sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance",ascending=False),
        edgecolor=('black'), linewidth=2, palette="colorblind")
plt.title('RFC Features importance', fontsize=15)
plt.tight_layout()
plt.savefig("rf_importance.png")


# In[ ]:


from sklearn.linear_model import LogisticRegression as LR

acc_lr = []
auc_lr = []
feature_coeff_lr = pd.DataFrame()



X_train, X_test, y_train, y_test = train_test_split(ff, yy, 
                                                        test_size=0.25,
                                                        stratify=yy,
                                                        shuffle=True)
    
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
    
lr = LR(C=2)
lr.fit(X_train, y_train)

y_pred_prob = lr.predict_proba(X_test)
y_pred = lr.predict(X_test)
acc_lr.append(accuracy_score(y_test, y_pred))
    
fpr, tpr, t = roc_curve(y_test, y_pred_prob[:,1])
auc_s = auc(fpr, tpr)
auc_lr.append(auc_s)
    
fold_coeff_lr = pd.DataFrame()
fold_coeff_lr["feature"] = ff.columns
fold_coeff_lr["coeff"] = np.abs(lr.coef_[0,:])

feature_coeff_lr = pd.concat([feature_coeff_lr, fold_coeff_lr], axis=0)


# In[ ]:


print(f"---------- Accuracy ------------ \n      Max: {np.max(acc_lr)} \n      Min: {np.min(acc_lr)} \n      Mean: {np.mean(acc_lr)}\n      Std: {np.std(acc_lr)}")

print(f"---------- AUC Score ------------ \n      Max: {np.max(auc_lr)} \n      Min: {np.min(auc_lr)} \n      Mean: {np.mean(auc_lr)}\n      Std: {np.std(auc_lr)}")


# In[ ]:


sns.set_style('whitegrid')

cols = (feature_coeff_lr[["feature", "coeff"]]
    .groupby("feature")
    .mean()
    .sort_values(by="coeff", ascending=False)[:30].index)

best_features = feature_coeff_lr.loc[feature_coeff_lr['feature'].isin(cols)]

plt.figure(figsize=(10,10))
sns.barplot(x="coeff", y="feature", data=best_features.sort_values(by="coeff",ascending=False),
        edgecolor=('black'), linewidth=2, palette="colorblind")
plt.title('Logistic Regression Absolute Value of Coefficients', fontsize=15)
plt.tight_layout()
plt.savefig("lr_importance.png")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




