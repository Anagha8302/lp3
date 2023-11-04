#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, k_means #For clustering
from sklearn.decomposition import PCA #Linear Dimensionality 


# In[17]:


import chardet

# Detect the encoding
with open("C:/Users/Owner/Desktop/datasets/sales_data_sample.csv", 'rb') as file:
    result = chardet.detect(file.read())

# Use the detected encoding
df = pd.read_csv("C:/Users/Owner/Desktop/datasets/sales_data_sample.csv", encoding=result['encoding'])


# In[18]:


df.head()


# In[19]:


df.describe()


# In[21]:


df.shape


# In[22]:


df.info()


# In[23]:


df.isnull().sum()


# In[24]:


df.dtypes


# In[25]:


df_drop = ['ADDRESSLINE1', 'ADDRESSLINE2', 'STATUS','POSTALCODE',
'CITY', 'TERRITORY', 'PHONE', 'STATE', 'CONTACTFIRSTNAME',
'CONTACTLASTNAME', 'CUSTOMERNAME', 'ORDERNUMBER']
df = df.drop(df_drop, axis=1)


# In[26]:


df.isnull().sum()


# In[27]:


df.dtypes


# In[28]:


df['COUNTRY'].unique()


# In[29]:


df['PRODUCTLINE'].unique()


# In[30]:


df['DEALSIZE'].unique()


# In[31]:


productline = pd.get_dummies(df['PRODUCTLINE']) 
Dealsize = pd.get_dummies(df['DEALSIZE'])


# In[32]:


df = pd.concat([df,productline,Dealsize], axis = 1)
df_drop = ['COUNTRY','PRODUCTLINE','DEALSIZE'] 
df = df.drop(df_drop, axis=1)
df['PRODUCTCODE'] = pd.Categorical(df['PRODUCTCODE']).codes
df.drop('ORDERDATE', axis=1, inplace=True) 
df.dtypes


# In[33]:


distortions = []
K = range(1,10)
for k in K:
 kmeanModel = KMeans(n_clusters=k)
 kmeanModel.fit(df)
 distortions.append(kmeanModel.inertia_) 


# In[34]:


plt.figure(figsize=(16,8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()


# In[35]:


X_train = df.values 
X_train.shape
(2823, 19)


# In[36]:


model = KMeans(n_clusters=3,random_state=2) 
model = model.fit(X_train) 
predictions = model.predict(X_train) 
unique,counts = np.unique(predictions,return_counts=True)
counts = counts.reshape(1,3)
counts_df =pd.DataFrame(counts,columns=['Cluster1','Cluster2','Cluster3'])
counts_df.head()


# In[37]:


pca = PCA(n_components=2) 
reduced_X =pd.DataFrame(pca.fit_transform(X_train),columns=['PCA1','PCA2'])
reduced_X.head()


# In[38]:


plt.figure(figsize=(14,10))
plt.scatter(reduced_X['PCA1'],reduced_X['PCA2'])


# In[39]:


model.cluster_centers_


# In[40]:


reduced_centers = pca.transform(model.cluster_centers_) 
reduced_centers


# In[41]:


plt.figure(figsize=(14,10))
plt.scatter(reduced_X['PCA1'],reduced_X['PCA2'])
plt.scatter(reduced_centers[:,0],reduced_centers[:,1],color='black',marker='x',s=300)


# In[42]:


reduced_X['Clusters'] = predictions 
reduced_X.head()


# In[43]:


plt.figure(figsize=(14,10))
plt.scatter(reduced_X[reduced_X['Clusters'] ==
0].loc[:,'PCA1'],reduced_X[reduced_X['Clusters'] ==
0].loc[:,'PCA2'],color='slateblue')
plt.scatter(reduced_X[reduced_X['Clusters'] ==
1].loc[:,'PCA1'],reduced_X[reduced_X['Clusters'] ==
1].loc[:,'PCA2'],color='springgreen')
plt.scatter(reduced_X[reduced_X['Clusters'] ==
2].loc[:,'PCA1'],reduced_X[reduced_X['Clusters'] ==
2].loc[:,'PCA2'],color='indigo')
plt.scatter(reduced_centers[:,0],reduced_centers[:,1],color='black',marker='x',s=300)


# In[ ]:




