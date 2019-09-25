#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.ticker as ticker
from sklearn import preprocessing
from matplotlib.ticker import NullFormatter
import itertools


# In[2]:


df=pd.read_csv('student-mat.csv')
df.head()


# In[9]:


x=df[['Medu','Fedu','famrel','freetime','goout','Dalc','Walc','health','absences','G1','G2']].values
x[0:5]


# In[5]:


y=df['G3'].values


# In[10]:


x=preprocessing.StandardScaler().fit(x).transform(x.astype(float))


# In[14]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=4)
print("Train is" , x_train.shape,y_train.shape)
print("test data is ",x_test.shape,y_train.shape)


# In[21]:


from sklearn.neighbors import KNeighborsClassifier 
from sklearn import metrics
ks=315
mean_acc=np.zeros((ks-1))
for n in range (1,ks):
    neigh=KNeighborsClassifier(n_neighbors=n).fit(x_train,y_train)
    yhat=neigh.predict(x_test)
    mean_acc[n-1]=metrics.accuracy_score(y_test,yhat)
mean_acc
    


# In[23]:


plt.plot(range(1,ks),mean_acc,'g')
plt.ylabel("Accuracy")
plt.xlabel("Number of Neighbors")
plt.show()


# In[26]:


print("The best accuracy is ", mean_acc.max()," with k = ",mean_acc.argmax()+1)


# In[ ]:




