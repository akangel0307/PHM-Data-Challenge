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
        dataframe = dataframe.drop(columns=['TIMESTAMP'])                  #'TIMESTAMP'数据是取均值的数据，没有意义，删掉
        dataframe_x = dataframe_x.append(dataframe,ignore_index=True)
    # dataframe_group_x = dataframe_x.groupby(['WAFER_ID','STAGE'])
    y_path = os.path.join(PATH, "CMP-"+stage+"-removalrate.csv")           #CMP-test-removalrate.csv
    dataframe_y = pd.read_csv(y_path)

    dataframe_y = dataframe_y.loc[dataframe_y['AVG_REMOVAL_RATE'] <= 1000] #大于1000的数据，可以认为是异常值，进行舍弃
    dataframe_y.hist('AVG_REMOVAL_RATE')                                   #对这一列绘制分布直方图
    # plt.hist(dataframe_y['AVG_REMOVAL_RATE'])                            #将这列数据绘制出来，这列数据明显的分为两段
    # plt.show()

    # print("dataframe_x.shape",dataframe_x.shape)
    # print("dataframe_y.shape", dataframe_y.shape)
    return dataframe_x, dataframe_y


def abstract_statistics(dataframe_x, dataframe_y, statistics=['mean','std','min','median','max']):
    # abstract statistics for virtual metrology
    # dataframe_x has dropped timestamps
    dataframe_group_x = dataframe_x.groupby(['WAFER_ID','STAGE'])                       # 对数据进行分组

    dataframe_statistics = dataframe_group_x.agg(statistics)                            # 计算分组后各个维度数据的相关统计量
    # print("dataframe_statistics",dataframe_statistics)
    # dataframe_statistics.to_csv("dataframe_statistics.csv", index=False, sep=',')     # 将上述相关统计量写入csv文件

    columns = dataframe_x.columns                                                       # 原始数据的列名
    dataframe_statistics.columns = generate_columns_name(columns, statistics)           # 为上述相关统计量数据生成新列名
    dataframe_statistics = pd.DataFrame(dataframe_statistics)
    dataframe_statistics.reset_index(inplace=True)                                      # 还原索引,填补空缺数据
    # dataframe_statistics.to_csv("dataframe_statistics_final.csv", index=False, sep=',') # 将上述相关统计量写入csv文件

    # data = pd.concat([dataframe_statistics, dataframe_y], ignore_index=True)
    data = pd.merge(dataframe_statistics, dataframe_y)
    # data.to_csv("data_final.csv", index=False, sep=',')
    return data


#函数功能：为相关统计量数据生成新列名
#输入参数：columns：特征数据，statistics：特征数据要计算的统计量
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
    train_data = train_data[train_data.columns[1:]].values         # 删掉WAFER_ID数据
    return train_data


#函数功能：PHM数据集进行三模态划分
#进行模态划分后，删掉STAGE数据
def split_data(data):
    splited_data = []
    for i in range(3):
        if i==0:
            idx = np.where(data[:, -1] > 120)                       # 模态1：y>120
            splited_data.append(np.squeeze(data[idx,1:], axis=0))

        elif i==1:                                                  # 模态2： y<120&stage='A'
            idx = np.where((data[:, -1] <= 120) & (data[:,0]=='A'))
            splited_data.append(np.squeeze(data[idx,1:], axis=0))

        elif i==2:                                                  # 模态3： y<120&stage='B'
            idx = np.where((data[:, -1] <= 120) & (data[:,0]=='B'))
            splited_data.append(np.squeeze(data[idx, 1:], axis=0))
    # print(splited_data)
    return splited_data

def split_data_label(data):
    x = data[:,:-1]
    y = data[:,-1]
    return x, y

if __name__ == "__main__":
    #分别生成训练数据集、测试数据集、验证数据集
    # train_data
    PATH = "D:\\repos\\data\\2016 PHM Data Challenge\\2016 PHM DATA CHALLENGE CMP DATA SET\\"
    stage = "training"
    stage_x = 'CMP-data\\'+stage
    train_data = load_data(PATH, stage, stage_x)
    np.save("./Processed data set/train_data.npy", train_data)

    # test_data
    PATH = "D:\\repos\\data\\2016 PHM Data Challenge\\2016 PHM DATA CHALLENGE CMP DATA SET\\"
    stage = "test"
    stage_x = 'CMP-data\\'+stage
    test_data =load_data(PATH, stage, stage_x)
    np.save("./Processed data set/test_data.npy", test_data)

    # validation_data
    PATH = "D:\\repos\\data\\2016 PHM Data Challenge\\2016 PHM DATA CHALLENGE CMP VALIDATION DATA SET\\"
    stage = "validation"
    stage_x = stage
    validation_data = load_data(PATH, stage, stage_x)
    np.save("./Processed data set/validation_data.npy", validation_data)