#!/usr/bin/env python
# coding: utf-8

# In[54]:


import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import numpy as np


# In[40]:


iris=load_iris()


# In[41]:


feature= pd.DataFrame(iris['data'],columns=iris['feature_names'])
target=pd.DataFrame(iris['target'],columns=['class'])


# In[42]:


data=pd.concat([feature,target],axis=1)
df=data[data['class'] !=2]
# 只允許出現類別為0或1的資料


# In[43]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("ggplot")
g=sns.FacetGrid(df,hue='class',size=5)
g.map(plt.scatter,"sepal length (cm)","sepal width (cm)")
g.add_legend()


# In[44]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[45]:


# 標準化
X=df.iloc[:,:2].values
y=df.iloc[:,4].values

sc=StandardScaler()
sc.fit(X)
X_std=sc.transform(X)


# In[46]:


from sklearn.linear_model import LogisticRegression
from matplotlib.colors import  ListedColormap


# In[47]:


# 訓練模型
lr=LogisticRegression(C=100.0,random_state=1)
#C=1/入
lr.fit(x_std,y)

print(lr.coef_)
print(lr.intercept_)
# intercept is 截距
# coef is 迴歸係數


# In[65]:


#定義決策區域
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


# In[66]:


#畫出目前模型的決策邊界
plot_decision_regions(X_std,y,classifer=lr)
plt.xlabel('sepal length[standardized]')
plt.ylabel('seapl width [standardized]')
plt.legend(loc='uper left')
plt.tight_layout()
#plt.savefig('images.png')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




