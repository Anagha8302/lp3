#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC,LinearSVC
from sklearn import metrics
from sklearn import preprocessing


# In[2]:


df=pd.read_csv("C:\\Users\\Owner\\Desktop\\datasets\\emails.csv")


# In[3]:


df.info()


# In[4]:


df.head()


# In[5]:


df.dtypes


# In[6]:


df=df.drop(['Email No.'],axis=1)


# In[7]:


df.isnull().sum()


# In[8]:


df.dtypes


# In[9]:


x=df.iloc[:,df.shape[1]-1]
y=df.iloc[:,-1]
x.shape,y.shape


# In[16]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.15)


# In[20]:


models={
    "K Nearest Neighbors":KNeighborsClassifier(n_neighbors=2),
    "Linear SVM":LinearSVC(random_state=8,max_iter=900000),
    "Polynomial SVM":SVC(kernel="poly",degree=2,random_state=8),
    "RBF SVM":SVC(kernel="rbf",random_state=8),
    "Sigmoid SVM":SVC(kernel="sigmoid",random_state=8)
}


# In[21]:


from sklearn import metrics
import numpy as np

for model_name, model in models.items():
    # Reshape the input data to ensure it's 2D
    x_train_reshaped = np.array(x_train).reshape(-1, 1)
    x_test_reshaped = np.array(x_test).reshape(-1, 1)

    # Fit the model to the training data
    model.fit(x_train_reshaped, y_train)
    
    # Make predictions on the test data
    y_pred = model.predict(x_test_reshaped)
    
    # Calculate and print the accuracy score
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print(f"Accuracy for {model_name} model: {accuracy}")


# In[19]:


df.describe()
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Absolute % Error:', metrics.mean_absolute_percentage_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R Squared (R²):', np.sqrt(metrics.r2_score(y_test, y_pred)))

# In[ ]:




