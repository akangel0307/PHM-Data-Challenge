import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

'''
PATH="D:\\repos\\data\\2016 PHM Data Challenge\\2016 PHM DATA CHALLENGE CMP DATA SET\\"
stage = "test"
stage_x = 'CMP-data\\'+stage
'''
def load_dataframe(PATH, stage, stage_x):
    # get x and y from corresponding dirs
    path = os.path.join(PATH, stage_x)                                     #路径名拼接，得到文件路径：PATH\stage_x
    dataframe_x = pd.DataFrame()
    for file_name in os.listdir(path):
        dataframe = pd.read_csv(os.path.join(path, file_name))
        dataframe = dataframe.drop(columns=['TIMESTAMP'])
        dataframe_x = dataframe_x.append(dataframe,ignore_index=True)
    # dataframe_group_x = dataframe_x.groupby(['WAFER_ID','STAGE'])
    y_path = os.path.join(PATH, "CMP-"+stage+"-removalrate.csv")           #CMP-test-removalrate.csv
    dataframe_y = pd.read_csv(y_path)

    dataframe_y = dataframe_y.loc[dataframe_y['AVG_REMOVAL_RATE'] <= 1000] #大于1000的数据，可以认为是异常值，进行舍弃
    dataframe_y.hist('AVG_REMOVAL_RATE')                                   #对这一列绘制分布直方图
    # plt.hist(dataframe_y['AVG_REMOVAL_RATE'])                            #将这列数据绘制出来，这列数据明显的分为两段
    # plt.show()
    return dataframe_x, dataframe_y

def merge_x_y(dataframe_x, dataframe_y):
    data = pd.merge(dataframe_x, dataframe_y)
    return data

def abstract_statistics(dataframe_x, dataframe_y, statistics=['mean','std','min','median','max']):
    # abstract statistics for virtual metrology
    # dataframe_x has dropped timestamps
    dataframe_group_x = dataframe_x.groupby(['WAFER_ID','STAGE'])

    dataframe_statistics = dataframe_group_x.agg(statistics)
    # dataframe_statistics = dataframe_statistics.drop(columns = [('AVG_REMOVAL_RATE','std'),('AVG_REMOVAL_RATE','min'),('AVG_REMOVAL_RATE','median'),('AVG_REMOVAL_RATE','max')])
    columns = dataframe_x.columns
    dataframe_statistics.columns = generate_columns_name(columns, statistics)
    dataframe_statistics = pd.DataFrame(dataframe_statistics)
    dataframe_statistics.reset_index(inplace=True)
    # data = pd.concat([dataframe_statistics, dataframe_y], ignore_index=True)
    data = pd.merge(dataframe_statistics, dataframe_y)
    return data

def generate_columns_name(columns, statistics):
    columns_list = []
    for column in columns:
        for statistic in statistics:
            if column not in ['MACHINE_ID','MACHINE_DATA','TIMESTAMP','WAFER_ID','STAGE']:  #过滤掉dataframe_statistics表中没有的列名
                columns_list.append(statistic + "_" + column)
    return columns_list

def load_data(PATH, stage, stage_x):
    dataframe_x, dataframe_y = load_dataframe(PATH, stage, stage_x)
    train_data = abstract_statistics(dataframe_x, dataframe_y)
    train_data = train_data[train_data.columns[2:]].values
    return train_data

def split_data(data, partitions=[50,100,165]):
    n = len(partitions)
    start = partitions[0]
    splited_data = []
    idx = np.where(data[:,-1]<=start)
    splited_data.append(np.squeeze(data[idx,:],axis=0))
    for i in range(1,n):
        end = partitions[i]
        idx = np.where(data[:,-1]<=start)
        splited_data.append(np.squeeze(data[idx,:], axis=0))
        start = end
    idx = np.where(data[:,-1]>start)
    splited_data.append(np.squeeze(data[idx,:],axis=0))
    return splited_data

def split_data_label(data):
    x = data[:,:-1]
    y = data[:,-1]
    return x, y

if __name__ == "__main__":
    # 在D盘新建文件夹(命名为repos)，将data文件夹放入到该文件夹中
    PATH = "D:\\repos\\data\\2016 PHM Data Challenge\\2016 PHM DATA CHALLENGE CMP DATA SET\\"
    stage = "test"
    stage_x = 'CMP-data\\'+stage
    test_data = load_data(PATH, stage, stage_x)
    print(test_data.shape)
    splited_data = split_data(test_data, [120])
    i = 0
    for data in  splited_data:
        print(i, data.shape)
        i += 1