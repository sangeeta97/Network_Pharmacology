#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re


# In[2]:


df1= pd.read_csv("edge.csv")


# In[3]:


df2= pd.read_csv("link_median_attach.csv")


# In[4]:


df3= pd.read_csv("link_median_jc.csv")


# In[5]:


df4= pd.read_csv("link_median_common.csv")


# In[6]:


df11= pd.concat([df2, df3, df4], sort= False)


# In[8]:


df11.drop_duplicates(subset= ['index'], inplace= True)


# In[31]:


df22= df11[df11.common != 0]


# In[12]:


from pandas import plotting

# for visualizations
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')


# In[14]:


df1['edge']= list(zip(df1['Unnamed: 0'], df1['Unnamed: 1']))


# In[16]:


df1.set_index(['edge'], inplace= True)


# In[18]:


df1= df1.drop(['Unnamed: 0', 'Unnamed: 1'], axis= 1)


# In[ ]:


def heatmap():
    plt.rcParams['figure.figsize'] = (15, 8)
    sns.heatmap(df1.corr(), cmap = 'Wistia', annot = True)
    plt.title('Heatmap for the edge Data', fontsize = 20)
    plt.xticks(rotation=45)
    plt.savefig("heatmap1.png")
    


# In[ ]:


def pairplot():
    plt.rcParams["axes.labelsize"] = 10
    sns.set_style("ticks", {"xtick.major.size": 3, "ytick.major.size": 3})
    sns.pairplot(df1)
    plt.xticks(rotation=45)
    plt.savefig('pairplot1.png')
    


# In[39]:


df22.rename({"index": "edge"}, axis= 1, inplace= True)


# In[40]:


df22.set_index(['edge'], inplace= True)


# In[42]:


df22.drop(['Unnamed: 0'], axis= 1, inplace= True)


# In[45]:


df1.drop(['weight'], axis= 1, inplace= True)


# In[49]:


df1.columns= ['attachment', 'common', 'jc', 'resource']


# In[50]:


dfall= pd.concat([df1, df22], keys= ['true_edge', 'false_edge'])


# In[51]:


dfall.reset_index(inplace= True)


# In[53]:


dfall.drop(['edge'], axis= 1, inplace= True)


# In[54]:


dfall.rename({'level_0': "edge"}, axis= 1, inplace= True)


# In[ ]:


def stripplot21():
    plt.rcParams['figure.figsize'] = (18, 7)
    sns.stripplot(dfall['edge'], dfall['attachment'], palette = 'Purples', size = 10)
    plt.title('edge type vs attachment Score', fontsize = 20)
    plt.savefig("stripplot1.png")


# In[ ]:


def stripplot22():
    plt.rcParams['figure.figsize'] = (18, 7)
    sns.stripplot(dfall['edge'], dfall['common'], palette = 'Purples', size = 10)
    plt.title('edge type vs common_neigh Score', fontsize = 20)
    plt.savefig("stripplot2.png")


# In[ ]:


def stripplot23():
    plt.rcParams['figure.figsize'] = (18, 7)
    sns.stripplot(dfall['edge'], dfall['jc'], palette = 'Purples', size = 10)
    plt.title('edge type vs jc Score', fontsize = 20)
    plt.savefig("stripplot3.png")


# In[ ]:


def stripplot24():
    plt.rcParams['figure.figsize'] = (18, 7)
    sns.stripplot(dfall['edge'], dfall['resource'], palette = 'Purples', size = 10)
    plt.title('edge type vs resource Score', fontsize = 20)
    plt.savefig("stripplot4.png")


# In[ ]:


def boxplot21():
    plt.rcParams['figure.figsize'] = (18, 7)
    sns.boxenplot(dfall['edge'], dfall['attachment'], palette = 'Blues')
    plt.title('edge type vs attachment Score', fontsize = 20)
    plt.savefig("boxplt1.png")


# In[ ]:


def boxplot22():
    plt.rcParams['figure.figsize'] = (18, 7)
    sns.boxenplot(dfall['edge'], dfall['common'], palette = 'Blues')
    plt.title('edge type vs common_neigh Score', fontsize = 20)
    plt.savefig("boxplt2.png")


# In[ ]:


def boxplot23():
    plt.rcParams['figure.figsize'] = (18, 7)
    sns.boxenplot(dfall['edge'], dfall['jc'], palette = 'Blues')
    plt.title('edge type vs jc Score', fontsize = 20)
    plt.savefig("boxplt3.png")


# In[ ]:


def boxplot24():
    plt.rcParams['figure.figsize'] = (18, 7)
    sns.boxenplot(dfall['edge'], dfall['resource'], palette = 'Blues')
    plt.title('edge type vs resource Score', fontsize = 20)
    plt.savefig("boxplt4.png")


# In[ ]:


sns.set(style="darkgrid")


# In[ ]:


def facegrid1():
    fg = sns.FacetGrid(data=dfall,hue='edge',height=5,aspect=1.5)
    fg.map(plt.scatter,'common','resource').add_legend()
    plt.savefig("facegrid1.png")
    


# In[ ]:


def facegrid2():
    fg = sns.FacetGrid(data=dfall,hue='edge',height=5,aspect=1.5)
    fg.map(plt.scatter,'common','jc').add_legend()
    plt.savefig("facegrid2.png")
    


# In[ ]:


def facegrid3():
    fg = sns.FacetGrid(data=dfall,hue='edge',height=5,aspect=1.5)
    fg.map(plt.scatter,'common','attachment').add_legend()
    plt.savefig("facegrid3.png")
    


# In[ ]:


x= dfall.drop(['edge'], axis=1)


# In[ ]:


def dendo():
    import scipy.cluster.hierarchy as sch
    
    dendrogram = sch.dendrogram(sch.linkage(x, method = 'ward'))
    plt.title('Dendrogam', fontsize = 20)
    plt.xlabel('all_edges')
    plt.ylabel('Ecuclidean Distance')
    plt.savefig("dendogram.png")
    
    


# In[ ]:


import multiprocessing
from multiprocessing import Process


# In[ ]:


if __name__=="__main__":
    p1=Process(target=heatmap)
    p2=Process(target=pairplot)
    p3=Process(target=stripplot21)
    p4=Process(target=stripplot22)
    p5=Process(target=stripplot23)
    p6=Process(target=stripplot24)
    p7=Process(target=boxplot21)
    p8=Process(target=boxplot22)
    p9=Process(target=boxplot23)
    p10=Process(target=boxplot24)
    p11=Process(target=facegrid1)
    p12=Process(target=facegrid2)
    p13=Process(target=facegrid3)
    p14=Process(target=dendo)
    
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





# In[ ]:




