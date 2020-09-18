from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import linear_model
from sklearn import svm
from sklearn.ensemble import GradientBoostingRegressor
from  sklearn.tree import ExtraTreeRegressor
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import  xgboost as  xgb



from sklearn.preprocessing import StandardScaler

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import data_utils

def try_different_model(regressor, X_train, X_test, y_train, y_test):
    modleName=str(regressor).split("(")            #提出出当前模型的名称
    modleName=modleName[0]

    regressor.fit(X_train, y_train)
    expected = y_test
    predicted = regressor.predict(X_test)
    
    # 绘制预测值图像和实值图像
    plt.figure(figsize=(15,3))
    # plt.scatter(np.arange(len(y_test)), expected, s=2, c='b', marker='.', label='true value')
    plt.plot(np.arange(len(y_test)),expected,color='blue',linewidth=1.0,marker = '.', linestyle='-',label='true value')
    plt.plot(np.arange(len(y_test)),predicted,color='red',linewidth=1.0,marker = '*', linestyle='-',label='predict value')
    plt.title(modleName+"  "+'prediction curve')
    plt.legend(loc='upper right')
    plt.savefig(modleName +".png")                                      # 保存模型图像
    plt.show()
    # 评价预测的准确性  
    print('MSE: ', mean_squared_error(expected, predicted))             # 均方误差，越小越好
    print('RMSE: ', np.sqrt(mean_squared_error(expected, predicted)))   # 根均方误差，越小越好
    print('MAE: ', mean_absolute_error(expected, predicted))            # 平均绝对误差，越小越好
    print('R^2: ', r2_score(expected, predicted))                       # r2 score 满分1，越接近1越好

##############  load train and test   #############
test_data = np.load("./Processed data set/test_data.npy")
train_data = np.load("./Processed data set/train_data.npy")
validation_data = np.load("./Processed data set/validation_data.npy")

########################### regressors config ###########################
#线性回归
LR = linear_model.LinearRegression()
# #CoReg半监督回归
# CoReg = coreg.Coreg(k1=3, k2=3, p1=2, p2=5, max_iters=100, pool_size=100, trials=1, verbose=False)
#KNN回归
KNN = KNeighborsRegressor(n_neighbors=15)
#GPR
GPR = GaussianProcessRegressor()
#PLS
PLS = PLSRegression(n_components=45)
#RF
RF = RandomForestRegressor()
#AdaBoost
Ada = AdaBoostRegressor()
#SVR
SVR = svm.SVR()
#GBDT
GBDT = GradientBoostingRegressor()
#ExtraTree
ExtraTree=ExtraTreeRegressor()

#Bagging回归，采用500个决策树进行组合
#sklearn的Bagging要求label为整形数据，但是label有float数据，bagging方法待实现
tree=DecisionTreeClassifier(criterion='entropy')
Bagging = BaggingClassifier(base_estimator=GBDT, n_estimators=500)
#xgboost
xgboost = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=200, silent=True, objective='reg:gamma')
#glmnetj-ElasticNet
ElasticNet=linear_model.ElasticNet()

#将上述模型组合,方便下面依次调用各个模型
models=[LR,KNN,GPR,PLS,RF,Ada,SVR,GBDT,ExtraTree,xgboost,ElasticNet]

partitions = [120]
splited_train_data = data_utils.split_data(train_data, partitions)
splited_test_data = data_utils.split_data(test_data, partitions)
num_partitions = len(partitions) + 1

for siglemodel in models:                     #依次调用各个分类模型
    modleName=str(siglemodel).split("(")      #当前预测模型的名称
    print("当前使用的模型"+"  "+modleName[0])
    for num in range(num_partitions):         #对数据进行分段
        print('set{}'.format(num))
        train_data = splited_train_data[num]
        X_train, y_train = data_utils.split_data_label(train_data)
        test_data = splited_test_data[num]
        X_test, y_test = data_utils.split_data_label(test_data)
        ss = StandardScaler()                 #Sklearn库函数，去均值和方差归一化
        X_train = ss.fit_transform(X_train)
        X_test = ss.transform(X_test)
        try_different_model(siglemodel, X_train, X_test, y_train, y_test)