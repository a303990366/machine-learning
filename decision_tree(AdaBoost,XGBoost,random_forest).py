#!/usr/bin/env python
# coding: utf-8

# In[51]:


def plot_decision_regions(X,y,classifer,test_idx=None,resolution=0.02):
    
    #setup marker generator and color map
    markers=('s','x','o','^','v')
    colors=('red','blue','lightgreen','gray','cyan')
    cmap=ListedColormap(colors[:len(np.unique(y))])
    
    #plot the decision surface
    x1_min,x1_max=X[:,0].min()-1,X[:,0].max()+1
    x2_min,x2_max=X[:,1].min(-1),X[:,1].max()+1
    xx1,xx2=np.meshgrid(np.arange(x1_min,x1_max,resolution),
                       np.arange(x2_min,x2_max,resolution))
    z=classifer.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    z=z.reshape(xx1.shape)
    plt.contourf(xx1,xx2,z,alpha=0.3,cmap=cmap)
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())
    
    for idx,cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y== cl,0],
                    y=X[y== cl,1],
                   alpha=0.8,
                   c=colors[idx],
                   marker=markers[idx],
                   label=cl,
                   edgecolor='black')
    #highlight test samples
    if test_idx:
        #plot all samples
        X_test,y_test=X[test_idx,:],y[test_idx]
        plt.scatter(X_test[:,0],
                    X_test[:,1],
                    c='',
                    edgecolor='blasck',
                    alpha=1.0,
                    linewidth=1,
                    marker='o',
                    s=100,
                    label='test set')


# In[5]:


import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt
from sklearn.datasets import load_wine
import seaborn as sns


# In[6]:


wine=load_wine()


# In[32]:


feature=pd.DataFrame(wine['data'][:,0],columns=['alcohol'])
feature1=pd.DataFrame(wine['data'][:,-2],columns=['od280/od315_of_diluted_wines'])
data=pd.concat([feature,feature1],axis=1)
target=pd.DataFrame(wine['target'],columns=['class'])
data=pd.concat([data,target],axis=1)
df=data[data['class']!=0]


# In[34]:


X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values


# In[39]:


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# In[40]:


le=LabelEncoder()
y=le.fit_transform(y)


# In[41]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1,stratify=y)


# In[50]:


#random forest

from sklearn.ensemble import RandomForestClassifier
from matplotlib.colors import  ListedColormap


# In[46]:


forest=RandomForestClassifier(criterion='entropy',
                             n_estimators=25,
                             random_state=1)
forest.fit(X_train,y_train)


# In[54]:


print("score:",forest.score(X_train,y_train))
print("score:",forest.score(X_test,y_test))


# In[52]:


plot_decision_regions(X,y,classifer=forest)


# In[55]:


#AdaBoost
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier


# In[99]:


tree=DecisionTreeClassifier(criterion='entropy',max_depth=1,random_state=1)
ada=AdaBoostClassifier(base_estimator=tree,
                      n_estimators=500,
                      learning_rate=0.1,
                      random_state=1)
ada=ada.fit(X_train,y_train)


# In[100]:


print("score:",ada.score(X_train,y_train))
print("score:",ada.score(X_test,y_test))


# In[101]:


plot_decision_regions(X,y,classifer=ada)


# In[115]:


#XGBoost
from xgboost import XGBClassifier


# In[116]:


xgbc=XGBClassifier()
xgbc.fit(X_train,y_train)


# In[117]:


print("score:",xgbc.score(X_train,y_train))
print("score:",xgbc.score(X_test,y_test))


# In[118]:


plot_decision_regions(X,y,classifer=xgbc)


# In[107]:


# decision tree
tree=DecisionTreeClassifier(criterion='gini',random_state=1,max_depth=5)
tree.fit(X_train,y_train)


# In[108]:


print("score:",tree.score(X_train,y_train))
print("score:",tree.score(X_test,y_test))


# In[109]:


plot_decision_regions(X,y,classifer=tree)


# In[120]:


data_dummies=pd.get_dummies(df,drop_first=True)
import graphviz
from sklearn.tree import export_graphviz
from pydotplus import graph_from_dot_data

dot_data=export_graphviz(tree,filled=True,
                        rounded=True,
                        class_names=data_dummies.iloc[:,-1].name,
                        out_file=None)
graph=graph_from_dot_data(dot_data)
#graph.write_png('tree.png')


# In[121]:


#graphviz 除了import之外，還要下載執行檔並配置
#參考來源:https://blog.csdn.net/qq_40304090/article/details/88594813
import sys
import os
os.environ["PATH"] += os.pathsep + 'D:/bin/'
sys.path.append("D:/bin")
#graph.write_png('1.png')


<a href="https://ibb.co/h28HqZW"><img src="https://i.ibb.co/nQkfWnc/1.png" alt="1" border="0"></a><br /><a target='_blank' href='https://imgbb.com/'>free upload pic</a><br />




