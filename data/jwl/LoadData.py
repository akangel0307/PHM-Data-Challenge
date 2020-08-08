import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

############################Loda Data#################################
def loadData_X(path):
    dataframe_x=pd.DataFrame()
    for file_name in os.listdir(path):
        dataframe = pd.read_csv(os.path.join(path, file_name))
        dataframe = dataframe.drop(columns=['TIMESTAMP'])
        dataframe_x = dataframe_x.append(dataframe,ignore_index=True)
    return  dataframe_x

def loadData_Y(path,fileName="CMP-test-removalrate.csv"):
    y_path = os.path.join(path, fileName)          # CMP-test-removalrate.csv
    dataframe_y = pd.read_csv(y_path)
    # dataframe_y['AVG_REMOVAL_RATE']数值分为两段，第一段是60~110，另一段是140~160
    # 所以大于1000的数据，可以认为是异常值，进行舍弃
    dataframe_y = dataframe_y.loc[dataframe_y['AVG_REMOVAL_RATE'] <= 1000]
    # plt.hist(dataframe_y['AVG_REMOVAL_RATE'])    # 绘制这一列数据的分布直方图
    # plt.show()
    return  dataframe_y

############################Feature Extraction#############################

# 函数功能：对数据集进行模态划分
def split_data(data, partitions=[50,100,165]):
    splited_data = []
    for i in range(len(partitions)):
        if i==0:
            index=np.where(data[:,-1]<=partitions[0])
        else:
            index = np.where(partitions[i-1]<data[:, -1] and data[:, -1] <= partitions[i])
        select_data = data[index]
        # np.squeeze函数:从数组的形状中删除单维度条目，即把shape中为1的维度去掉
        split_data.append(np.squeeze(select_data,axis=0))
    return splited_data

# 函数功能：提取数据集的特征数据和样本标签
def split_data_label(data):
    feature = data[:,:-1]
    label   = data[:,-1]
    return feature, label


############################Save Document#############################

#函数功能：将数据集文件另存为模型文件(.npy)
def save_document(data,file_path,file_name):
    path=os.path.join(file_path, file_name)
    np.save(path,data)

if __name__ == "__main__":

    #加载原始数据集
    data_X_path="./2016 PHM Data Challenge/2016 PHM DATA CHALLENGE CMP DATA SET/CMP-data/test"
    data_Y_path="./2016 PHM Data Challenge/2016 PHM DATA CHALLENGE CMP DATA SET"
    dataframe_x=loadData_X(data_X_path)
    dataframe_y=loadData_Y(data_Y_path,fileName="CMP-test-removalrate.csv")
    print(dataframe_x.shape)
    print(dataframe_y.shape)

    # 对数据集进行特征处理

    # 划分训练集、测试集、交叉验证集