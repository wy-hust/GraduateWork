import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import math

def read_data(file_path):
    data = pd.read_csv(file_path, usecols=[2], header=0,nrows=14401)
    data.fillna(method='ffill', inplace=True) #按上一行补全空白行
    data = data.values.astype('float32')
    return data

def dataLoad(data):
    

    # 数据集划分
    train_data_x = data[:-24]
    train_data_y = data[1:-23]


    # 创建数据加载器
    train_dataset = MultiTaskDataset(train_data_x,train_data_y)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
    return train_loader
# 多任务学习数据集
class MultiTaskDataset(Dataset):
    def __init__(self, data_x,data_y):
        self.data_x = data_x
        self.data_y = data_y
        

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, index):
        x = self.data_x[index]
        y = self.data_y[index]
        return np.array(x), np.array(y)




# 定义模型
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_tasks):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_tasks = num_tasks
        self.lstm = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, num_tasks)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        out, _ = self.lstm(x)
        out = self.fc1(out[:, -1, :])
        out = self.fc2(out)
        return out
def train(train_loader):
    # 超参数
    num_epochs = 20
    learning_rate = 0.01
    # 初始化模型
    model = GRU(input_size=1, hidden_size=64, num_layers=2, num_tasks=1)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 模型训练
    MinTrainLoss = 1e10
    for epoch in range(num_epochs):
        train_loss = 0
        for i,(inputs, labels) in enumerate(train_loader):
            inputs = inputs.reshape(-1, 1, 1)
            labels = labels.reshape(-1,1)
            outputs= model(inputs)
            loss = criterion(outputs,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        if train_loss<MinTrainLoss:
            torch.save(model.state_dict(),"single_model.pth")
        # 打印训练信息
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(train_loader):.4f}')


def test(data):
    model = GRU(input_size=1, hidden_size=64, num_layers=2, num_tasks=1)
    model.load_state_dict(torch.load("single_model.pth"))
    eval_x = data[-25:-1]
    eval_y = data[-24:]
    eval_dataset = MultiTaskDataset(eval_x,eval_y)
    eval_loader = DataLoader(eval_dataset,batch_size=1,shuffle = False)
    with torch.no_grad():
        miss_avg1 = 0
        MAE_avg = 0
        RMSE_avg = 0
        for i,(inputs, labels) in enumerate(eval_loader):
            inputs = inputs.reshape(-1, 1, 1)
            outputs = model(inputs)
            miss1 =abs(outputs[0][0].item()-eval_y[i])/eval_y[i]
            mae =  abs(outputs[0][0].item()-eval_y[i])
            rmse = (outputs[0][0].item()-eval_y[i])*(outputs[0][0].item()-eval_y[i])
            miss_avg1 += miss1
            MAE_avg += mae
            RMSE_avg += rmse
        print(f'MAPE_avg: {miss_avg1[0]/len(eval_loader):.4f}')
        print(f'MAE_avg: {mae[0]/len(eval_loader):.4f}')
        print(f'RMSE_avg: {math.sqrt(rmse[0]/len(eval_loader)):.4f}')
        

if __name__ == '__main__':
    data = read_data('data\data13.csv')
    train_loader = dataLoad(data)
    train(train_loader)
    test(data)
