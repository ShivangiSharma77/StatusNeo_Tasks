#!/usr/bin/env python
# coding: utf-8

# # Breast Cancer Data Cleaning

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


df = pd.read_csv('data.csv')


# In[6]:


df.shape


# In[7]:


df.head()


# In[8]:


df.info()


# In[9]:


df.columns


# In[10]:


#Checking for duplicated values in dataset
df.duplicated().sum()
#there are no duplicate values


# In[12]:


#Checking for null values in dataset
df.isnull().sum()
#there are no null records


# In[13]:


df = df.drop(['Unnamed: 32', 'id'], axis=1)
#We wont be needing the "Unnamed" column and "id" column so we drop it


# In[15]:


df.shape


# In[16]:


df['diagnosis'].value_counts()


# In[17]:


#The diagnosis column is in object form.
#Replace M with 1 (1 = cancerous)
#        B with 0 (0 = cancerous)


# In[18]:


df['diagnosis']= df['diagnosis'].replace('M', 1)
df['diagnosis']= df['diagnosis'].replace('B', 0)


# In[23]:


df.head()


# In[24]:


#There are 31 columns in the dataset, out of these columns many can be of no use to determine the diagnosis, hence we can find the corelation and determine the relationship between diagnosis and the other columns.


# In[26]:


corr = df.corr()


# In[29]:


plt.figure(figsize=(30,15))
sns.heatmap(corr, annot = True)
plt.show()


# In[30]:


corr[abs(corr['diagnosis']) > 0.59].index


# In[31]:


df = df[['diagnosis', 'radius_mean', 'perimeter_mean', 'area_mean',
       'compactness_mean', 'concavity_mean', 'concave points_mean',
       'radius_worst', 'perimeter_worst', 'area_worst', 'compactness_worst',
       'concavity_worst', 'concave points_worst']]


# In[32]:


df.shape


# In[33]:


df.head()


# In[ ]:




