from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import linear_model

from sklearn.preprocessing import StandardScaler

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import data_utils

def try_different_model(regressor, X_train, X_test, y_train, y_test):
    regressor.fit(X_train, y_train)
    expected = y_test
    predicted = regressor.predict(X_test)
    
    # 绘制预测值图像和实值图像
    plt.figure(figsize=(15,3))
    # plt.scatter(np.arange(len(y_test)), expected, s=2, c='b', marker='.', label='true value')
    plt.plot(np.arange(len(y_test)),expected,color='blue',linewidth=1.0,marker = '.', linestyle='-',label='true value')
    plt.plot(np.arange(len(y_test)),predicted,color='red',linewidth=1.0,marker = '*', linestyle='-',label='predict value')
    plt.title('prediction curve')
    plt.legend(loc='upper right') 
    plt.show()
    # 评价预测的准确性  
    print('MSE: ', mean_squared_error(expected, predicted))  # 均方误差，越小越好
    print('RMSE: ', np.sqrt(mean_squared_error(expected, predicted)))  # 根均方误差，越小越好
    print('MAE: ', mean_absolute_error(expected, predicted))  # 平均绝对误差，越小越好
    print('R^2: ', r2_score(expected, predicted))  # r2 score 满分1，越接近1越好

##############  load train and test   #############
# PATH = "D:\\repos\\data\\2016 PHM Data Challenge\\2016 PHM DATA CHALLENGE CMP DATA SET\\"
# stage = "test"
# stage_x = 'CMP-data\\'+stage
# test_data = data_utils.load_data(PATH, stage, stage_x)
# np.save("D:\\repos\\data\\jwl\\data\\test_data.npy", test_data)
test_data = np.load("D:\\repos\\data\\jwl\\data\\test_data.npy")

# PATH = "D:\\repos\\data\\2016 PHM Data Challenge\\2016 PHM DATA CHALLENGE CMP DATA SET\\"
# stage = "training"
# stage_x = 'CMP-data\\'+stage
# train_data = data_utils.load_data(PATH, stage, stage_x)
# np.save("D:\\repos\\data\\jwl\\data\\train_data.npy", train_data)
train_data = np.load("D:\\repos\\data\\jwl\\data\\train_data.npy")

# PATH = "D:\\repos\\data\\2016 PHM Data Challenge\\2016 PHM DATA CHALLENGE CMP VALIDATION DATA SET\\"
# stage = "validation"
# stage_x = stage
# validation_data = data_utils.load_data(PATH, stage, stage_x)
# np.save("D:\\repos\\data\\jwl\\data\\validation_data.npy", validation_data)
validation_data = np.load("D:\\repos\\data\\jwl\\data\\validation_data.npy")
########################### regressors config ###########################
#线性回归
LR = linear_model.LinearRegression()
# #CoReg半监督回归
# CoReg = coreg.Coreg(k1=3, k2=3, p1=2, p2=5, max_iters=100, pool_size=100, trials=1, verbose=False)
#KNN回归
KNN = KNeighborsRegressor(n_neighbors=3)
#GPR
GPR = GaussianProcessRegressor()
#PLS
PLS = PLSRegression(n_components=45)
#RF
RF = RandomForestRegressor()
#ADA
ADA = AdaBoostRegressor()

partitions = [120]
splited_train_data = data_utils.split_data(train_data, partitions)
splited_test_data = data_utils.split_data(test_data, partitions)
num_partitions = len(partitions) + 1
for num in range(num_partitions):
    print('set{}'.format(num))
    train_data = splited_train_data[num]
    X_train, y_train = data_utils.split_data_label(train_data)
    test_data = splited_test_data[num]
    X_test, y_test = data_utils.split_data_label(test_data)
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)
    try_different_model(ADA, X_train, X_test, y_train, y_test)