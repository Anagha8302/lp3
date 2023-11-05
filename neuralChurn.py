#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv("C:\\Users\\Owner\\Desktop\\datasets\\Churn_Modelling.csv")
df


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.describe()


# In[6]:


df.info()


# In[7]:


df.isnull().sum()


# In[8]:


df.columns


# In[9]:


df.drop(columns=['RowNumber', 'CustomerId', 'Surname'],axis=1)


# In[10]:


df.head()


# In[11]:


def visualization(x, y, xlabel):
    plt.figure(figsize=(10,5))
    plt.hist([x, y], color=['red', 'green'], label = ['exit', 'not_exit'])
    plt.xlabel(xlabel,fontsize=20)
    plt.ylabel("No. of customers", fontsize=20)
    plt.legend()
df_churn_exited = df[df['Exited']==1]['Tenure']
df_churn_not_exited = df[df['Exited']==0]['Tenure']
visualization(df_churn_exited, df_churn_not_exited, "Tenure")
df_churn_exited2 = df[df['Exited']==1]['Age']
df_churn_not_exited2 = df[df['Exited']==0]['Age']
visualization(df_churn_exited2, df_churn_not_exited2, "Age")


# In[12]:


x = df[['CreditScore','Gender','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary']]
states = pd.get_dummies(df['Geography'],drop_first = True)
gender = pd.get_dummies(df['Gender'],drop_first = True)
df = pd.concat([df,gender,states], axis = 1)


# In[13]:


df.head()


# In[15]:


x = df[['CreditScore','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary','Male','Germany','Spain']]
y = df['Exited']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.30)


# In[16]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
x_train
x_test


# In[17]:


x_test.shape


# In[29]:


from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score,accuracy_score,classification_report
from sklearn.metrics import ConfusionMatrixDisplay


# In[19]:


classifier = MLPClassifier(hidden_layer_sizes=(100, 50, 100), activation='relu', max_iter=1000)


# In[20]:


classifier.fit(x_train, y_train)


# In[21]:


y_pred = classifier.predict(x_test)


# In[22]:


cf_matrix=confusion_matrix(y_test,y_pred)
print(cf_matrix)


# In[23]:


ConfusionMatrixDisplay.from_predictions(y_test, y_pred)


# In[24]:


acc = accuracy_score(y_test, y_pred)
print(acc)


# In[25]:


ps = precision_score(y_test, y_pred)
print(ps)


# In[26]:


rs = recall_score(y_test, y_pred)
print(rs)


# In[27]:


f1s = f1_score(y_test, y_pred)
print(f1s)


# In[28]:


error_rate = 1 - acc
print(error_rate)


# In[30]:


cr = classification_report(y_test, y_pred)
print(cr)


# In[ ]:




