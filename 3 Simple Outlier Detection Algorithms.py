#!/usr/bin/env python
# coding: utf-8

# In[37]:


import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd


# In[2]:


X = np.array([1,2,4,6,8,10,12])
y = np.array([23,27,30,34,40,44,48])


# In[3]:


fig = px.scatter(X,y,trendline='ols',labels={'index':'y'})
fig.update_layout(title='OLS with no Outliers',title_x=0.5)


# In[4]:


X = np.array([1,2,22,4,6,8,10,12])
y = np.array([23,27,40,30,34,40,44,48])


# In[5]:


fig = px.scatter(X,y,trendline='ols',labels={'index':'y'})
fig.update_layout(title='OLS with an Outlier',title_x=0.5)


# In[6]:


# Data with no outliers
np.array([35,20,32,40,46,45]).mean()


# In[8]:


# Data with 2 outliers
np.array([1,35,20,32,40,46,45,4500]).mean()


# <h1>Solution 1: DBSCAN</h1>

# In[29]:


from sklearn.cluster import DBSCAN 
from sklearn.datasets import make_moons


X, y = make_moons(n_samples=1000, noise=0.05)

dbscan = DBSCAN(eps=0.2, min_samples=5)
dbscan.fit(X)


# In[30]:


dbscan.labels_


# In[31]:


from sklearn.neighbors import KNeighborsClassifier


# In[32]:


knn = KNeighborsClassifier(n_neighbors=50)
knn.fit(dbscan.components_,dbscan.labels_[dbscan.core_sample_indices_])


# In[33]:


X_new = np.array([[-0.5, 0], [0, 0.5], [1, -0.1], [2, 1]])
knn.predict(X_new)


# In[34]:


y_dist,y_pred_idx = knn.kneighbors(X_new,n_neighbors=1)
y_pred = dbscan.labels_[dbscan.core_sample_indices_][y_pred_idx]
y_pred[y_dist > 0.2] = -1
y_pred.ravel()


# <h1>Solution 2: IsolationForest</h1>

# In[35]:


from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error


# In[38]:


url='https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
df = pd.read_csv(url, header=None)
data = df.values
# split into input and output elements
X, y = data[:, :-1], data[:, -1]


# In[39]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X,y)
mean_absolute_error(lr.predict(X),y)


# In[40]:


iso = IsolationForest(contamination='auto',random_state=42)


# In[41]:


y_pred = iso.fit_predict(X,y)
mask = y_pred != -1


# In[42]:


X,y = X[mask,:],y[mask]


# In[43]:


lr.fit(X,y)
mean_absolute_error(lr.predict(X),y)


# In[44]:


import matplotlib.pyplot as plt
import seaborn as sns
X = np.array([45,56,78,34,1,2,67,68,87,203,-200,-150])
y = np.array([1,1,0,0,1,0,1,1,0,0,1,1])


# In[45]:


sns.boxplot(X)
plt.show()


# In[46]:


X = X[(X < 150) & (X > -50)]
sns.boxplot(X)
plt.show()


# In[52]:


from collections import Counter

def detect_outliers(df, n, features):
    # list to store outlier indices
    outlier_indices = []
    # iterate over features(columns)
    for col in features:
        # Get the 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # Get the 3rd quartile (75%)
        Q3 = np.percentile(df[col], 75)
        # Get the Interquartile range (IQR)
        IQR = Q3 - Q1
        # Define our outlier step
        outlier_step = 1.5 * IQR
        
        
       # Determine a list of indices of outliers
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index
       # append outlier indices for column to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
       # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)
    
    return multiple_outliers


# detect outliers from list of features
list_of_features = ['x1', 'x2']
# params dataset, number of outliers for rejection, list of features 
#Outliers_to_drop = detect_outliers(dataset, 2, list_of_features)


# In[ ]:




