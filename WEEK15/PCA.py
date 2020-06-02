#!/usr/bin/env python
# coding: utf-8

# In[175]:


from sklearn.decomposition import PCA
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier


# In[210]:


data=pd.read_csv('heart.csv')


# In[217]:


X=data.iloc[:,:-1].values
y=data.iloc[:,-1].values


# In[220]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,stratify=y)


# In[221]:


Std=StandardScaler()
x_std=Std.fit_transform(X_train)
X_std1=Std.transform(X_test)


# In[225]:


pca=PCA(n_components='mle')
pca.fit(x_std)
X_std_pca=pca.transform(x_std)#train_data
X_std1_pca=pca.transform(X_std1)#test_data


# In[226]:


rf=RandomForestClassifier(criterion='entropy',
                             n_estimators=100,
                             random_state=1)
rf.fit(X_std_pca,y_train)


# In[227]:


print("score:",rf.score(X_std_pca,y_train))
print("score:",rf.score(X_std1_pca,y_test))


# In[ ]:




