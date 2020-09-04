# README

## Data folders

- 2016 PHM Data Challenge文件夹：存放原始数据集
- Processed data set文件夹：存放做过特征选择处理的数据集

- model predict image文件夹：存放相关模型的预测走势图




## Code files

- data_utils.py：对数据集进行特征处理
- regression.py：对数据集进行建模处理。所采用的模型包括但不限于：LR,KNN,GPR,PLS,RF,Ada,SVR,GBDT,ExtraTree,xgboost,ElasticNet



## Requirements

- Python 3

- annaconda

- xgboost

## Experimental Results

参数声明

set0---dataframe_y['AVG_REMOVAL_RATE']<120

set1---dataframe_y['AVG_REMOVAL_RATE']>=120



