# 範例:

### from sklearn.model_selection import cross_val_score
### scores=cross_val_score(knn,X_train,y_train,cv=10,scoring='accuracy')

# 參數解釋:
### knn為需要進行交叉驗證的模型
### X_train為訓練集的訓練資料
### y_train為訓練集的預測資料
### cv為要對資料進行幾次的拆分，一般約為10次
### scoring為以甚麼方式計算分數

# 使用交叉驗證的用意:
### 可以用來進行參數的選擇
### 可以用來進行模型的選擇
### 可以用來調整模型參數

# 資料來源:
### https://blog.csdn.net/weixin_38536057/article/details/78702564
