import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import math
import matplotlib.pyplot as plt
def read_data(file_path):
    data = pd.read_csv(file_path, usecols=[2], header=0,nrows=14401)
    data.fillna(method='ffill', inplace=True) #按上一行补全空白行
    data = data.values.astype('float32')
    return data

def dataLoad(data1,data2):
    assert data1.shape[0] == data2.shape[0], "数据集1和2长度不一致"

    # 数据集划分
    train_data1_x = data1[:-24]
    train_data1_y = data1[1:-23]
    train_data2_x = data2[:-24]
    train_data2_y = data2[1:-23]

    # 创建数据加载器
    train_dataset = MultiTaskDataset(train_data1_x,train_data1_y, train_data2_x,train_data2_y)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
    return train_loader
# 多任务学习数据集
class MultiTaskDataset(Dataset):
    def __init__(self, data1_x,data1_y, data2_x,data2_y):
        self.data1_x = data1_x
        self.data1_y = data1_y
        self.data2_x = data2_x
        self.data2_y = data2_y

    def __len__(self):
        return len(self.data1_x)

    def __getitem__(self, index):
        x = [self.data1_x[index], self.data2_x[index]]
        y = [self.data1_y[index], self.data2_y[index]]
        return np.array(x), np.array(y)




# 定义模型
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_tasks):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_tasks = num_tasks
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, num_tasks)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc1(out[:, -1, :])
        out = self.fc2(out)
        return out
def train(train_loader):
    # 超参数
    num_epochs = 20
    #batch_size = 16
    learning_rate = 0.01
    # 初始化模型
    model = LSTM(input_size=2, hidden_size=64, num_layers=2, num_tasks=2)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 模型训练
    MinTrainLoss = 1e10
    for epoch in range(num_epochs):
        train_loss = 0
        for i,(inputs, labels) in enumerate(train_loader):
            inputs = inputs.reshape(-1, 1, 2)
            labels = labels.reshape(-1,2)
            outputs= model(inputs)
            loss = criterion(outputs,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        if train_loss<MinTrainLoss:
            torch.save(model.state_dict(),"baseline_model.pth")
        # 打印训练信息
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(train_loader):.4f}')


def test(data1,data2):
    model = LSTM(input_size=2, hidden_size=64, num_layers=2, num_tasks=2)
    model.load_state_dict(torch.load("baseline_model.pth"))
    eval_x1 = data1[-25:-1]
    eval_x2 = data2[-25:-1]
    eval_y1 = data1[-24:]
    eval_y2 = data2[-24:]
    eval_dataset = MultiTaskDataset(eval_x1,eval_y1,eval_x2,eval_y2)
    eval_loader = DataLoader(eval_dataset,batch_size=1,shuffle = False)
    with torch.no_grad():
        miss_avg1 = 0
        miss_avg2 = 0
        MAE_avg1 = 0
        RMSE_avg1 = 0
        MAE_avg2 = 0
        RMSE_avg2 = 0
        output1 = []
        output2 = []
        for i,(inputs, labels) in enumerate(eval_loader):
            inputs = inputs.reshape(-1, 1, 2)
            outputs = model(inputs)
            output1.append(outputs[0][0].item())
            output2.append(outputs[0][0].item())
            miss1 =abs(outputs[0][0].item()-eval_y1[i])/eval_y1[i]
            miss2 =abs(outputs[0][1].item()-eval_y2[i])/eval_y2[i]
            miss_avg1 += miss1
            miss_avg2 += miss2
            
            mae1 =  abs(outputs[0][0].item()-eval_y1[i])
            rmse1 = (outputs[0][0].item()-eval_y1[i])*(outputs[0][0].item()-eval_y1[i])
            MAE_avg1 += mae1
            RMSE_avg1 += rmse1

            mae2 =  abs(outputs[0][1].item()-eval_y2[i])
            rmse2 = (outputs[0][1].item()-eval_y2[i])*(outputs[0][1].item()-eval_y2[i])
            MAE_avg2 += mae2
            RMSE_avg2 += rmse2
        print(f'MAPE_avg1: {miss_avg1[0]/len(eval_loader):.4f}')
        print(f'MAE_avg1: {mae1[0]/len(eval_loader):.4f}')
        print(f'RMSE_avg1: {math.sqrt(rmse1[0]/len(eval_loader)):.4f}')
        print(f'MAPE_avg2: {miss_avg2[0]/len(eval_loader):.4f}')
        print(f'MAE_avg2: {mae2[0]/len(eval_loader):.4f}')
        print(f'RMSE_avg2: {math.sqrt(rmse2[0]/len(eval_loader)):.4f}')
        plt.plot(output1)
        plt.plot(eval_y1)
        plt.show()
        plt.plot(output2)
        plt.plot(eval_y2)
        plt.show()


if __name__ == '__main__':
    data1 = read_data('data\data5.csv')
    data2 = read_data('data\data13.csv')
    train_loader = dataLoad(data1,data2)
    train(train_loader)
    test(data1,data2)
