# README

## Data folders

|            文件夹             |               相关描述               |
| :---------------------------: | :----------------------------------: |
| 2016 PHM Data Challenge、data |    这两个文件夹均是存放原始数据集    |
|      Processed data set       |     存放做过特征选择处理的数据集     |
|      model predict image      |       存放相关模型的预测走势图       |
|   mode division experiment    | 存放对数据进行三模态划分后的相关文件 |




## Code files

- data_utils.py：对数据集进行特征处理
- regression.py：对数据集进行建模处理。所采用的模型包括但不限于：LR,KNN,GPR,PLS,RF,Ada,SVR,GBDT,ExtraTree,xgboost,ElasticNet
- SVR.py：对SVR模型的相关超参数进行试验



## Requirements

- Python 3

- annaconda

- xgboost



## Experimental Results

- 两模态划分实验

| set0 | dataframe_y['AVG_REMOVAL_RATE']<120  |
| :--: | :----------------------------------: |
| set1 | dataframe_y['AVG_REMOVAL_RATE']>=120 |



```python
python regression.py
```

- 三模态划分

相关code文件在mode division experiment文件夹

| set0 |      dataframe_y['AVG_REMOVAL_RATE']>120      |
| :--: | :-------------------------------------------: |
| set1 | dataframe_y['AVG_REMOVAL_RATE']<120&stage='A' |
| set2 | dataframe_y['AVG_REMOVAL_RATE']<120&stage='B' |



```python
python SVR.py
```

