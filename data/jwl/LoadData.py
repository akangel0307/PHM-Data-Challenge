import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

############################loda data#############################
def loadData_X(path):
    dataframe_x=pd.DataFrame()
    for file_name in os.listdir(path):
        dataframe = pd.read_csv(os.path.join(path, file_name))
        dataframe = dataframe.drop(columns=['TIMESTAMP'])
        dataframe_x = dataframe_x.append(dataframe,ignore_index=True)
    return  dataframe_x


def loadData_Y(path):
    dataframe_x=pd.DataFrame()
    for file_name in os.listdir(path):
        dataframe = pd.read_csv(os.path.join(path, file_name))
        dataframe = dataframe.drop(columns=['TIMESTAMP'])
        dataframe_x = dataframe_x.append(dataframe,ignore_index=True)
    return  dataframe_x




if __name__ == "__main__":

    #加载数据集
    data_X_path="./2016 PHM Data Challenge/2016 PHM DATA CHALLENGE CMP DATA SET/CMP-data/test"
    dataframe_x=loadData_X(data_X_path)