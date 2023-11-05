#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,classification_report, ConfusionMatrixDisplay


# In[3]:


df=pd.read_csv("C:\\Users\\rakhi\\Desktop\\Datasets\\emails.csv")
df


# In[4]:


x = df.drop(['Email No.', 'Prediction'], axis = 1)
y = df['Prediction']


# In[5]:


x


# In[6]:


y


# In[7]:


x.dtypes


# In[8]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state =42)


# In[9]:


sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)


# In[10]:


knn = KNeighborsClassifier(n_neighbors=5)  
knn.fit(x_train,y_train)
knn_predictions = knn.predict(x_test)
knn_accuracy = accuracy_score(y_test, knn_predictions)


# In[11]:


ConfusionMatrixDisplay.from_predictions(y_test, knn_predictions)


# In[12]:


print(classification_report(y_test, knn_predictions))


# In[13]:


svm = SVC(kernel='poly')
svm.fit(x_train, y_train)
svm_predictions = svm.predict(x_test)
svm_accuracy = accuracy_score(y_test, svm_predictions)


# In[14]:


ConfusionMatrixDisplay.from_predictions(y_test, svm_predictions)


# In[15]:


print(classification_report(y_test, svm_predictions))


# In[16]:


print("K-Nearest Neighbors Accuracy:", knn_accuracy)
print("Support Vector Machine Accuracy:", svm_accuracy)


# In[ ]:




