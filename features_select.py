#!/usr/bin/env python
# coding: utf-8

# In[302]:


from sklearn.datasets import load_iris
# from sklearn.datasets import load_wine
# from sklearn.datasets import load_boston
import pandas as pd


# In[316]:


from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# In[326]:


from sklearn.feature_selection import RFE

iris=load_iris()
#用RFE,返回特徵選擇後的資料
#參數estimator裡放機器學習模型
#參數n_feature_to_select為要選擇的特徵個數
for i in range(1,5):
    score=RFE(estimator=DecisionTreeRegressor(),n_features_to_select=i)
    X_t=score.fit_transform(iris.data,iris.target)

    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target,
                                                                test_size=0.25, random_state=0)
    X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X_t,iris.target,
                                                                test_size=0.25, random_state=0)


    t1=DecisionTreeRegressor().fit(X_train,y_train)
    t2=DecisionTreeRegressor().fit(X_train_t,y_train_t)
    print(i,end=':')
    print("Original DataSet: test score=%s" % (t1.score(X_test, y_test)))
    print("Selected DataSet: test score=%s" % (t2.score(X_test_t, y_test_t)))
    print('-------------')


# In[324]:


print(score.ranking_)
print(score.support_)


# In[325]:


list(zip(iris.feature_names,score.ranking_))

