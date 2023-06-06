import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

def read_data(file_path):
    data = pd.read_csv(file_path, usecols=[2], header=0,nrows=14401)
    data.fillna(method='ffill', inplace=True) #按上一行补全空白行
    data = data.values.astype('float32')
    return data

data1 = read_data('data\data4.csv')
data2 = read_data('data\data19.csv')
data1 = data1[:500]
data2 = data2[:500]
plt.plot(data1, label='user 4')
plt.plot(data2, label='user 19')
plt.title("Time-PowerLoad")
plt.xlabel("Time")
plt.ylabel("PowerLoad")

plt.legend()
plt.show()