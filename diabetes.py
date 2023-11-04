#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics


# In[3]:


df=pd.read_csv("C:\\Users\\Owner\\Desktop\\datasets\\diabetes.csv")


# In[4]:


df.columns


# In[5]:


df.isnull().sum()


# In[6]:


X = df.drop('Outcome',axis = 1)
y = df['Outcome']
from sklearn.preprocessing import scale
X = scale(X)


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =
0.3, random_state = 42)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("Confusion matrix: ")
cs = metrics.confusion_matrix(y_test,y_pred)
print(cs)


# In[8]:


print("Acccuracy ",metrics.accuracy_score(y_test,y_pred))


# In[9]:


total_misclassified = cs[0,1] + cs[1,0]
print(total_misclassified)
total_examples = cs[0,0]+cs[0,1]+cs[1,0]+cs[1,1]
print(total_examples)
print("Error rate",total_misclassified/total_examples)
print("Error rate ",1-metrics.accuracy_score(y_test,y_pred))


# In[10]:


print("Precision score",metrics.precision_score(y_test,y_pred))


# In[11]:


print("Recall score ",metrics.recall_score(y_test,y_pred))


# In[12]:


print("Classification report",metrics.classification_report(y_test,y_pred))


# In[ ]:




