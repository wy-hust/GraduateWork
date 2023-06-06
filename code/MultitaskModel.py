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

def dataLoad(data1,data2,batch_size):
    assert data1.shape[0] == data2.shape[0], "数据集1和2长度不一致"

    # 数据集划分
    train_data1_x = data1[:-24]
    train_data1_y = data1[1:-23]
    train_data2_x = data2[:-24]
    train_data2_y = data2[1:-23]

    # 创建数据加载器
    train_dataset = MultiTaskDataset(train_data1_x,train_data1_y, train_data2_x,train_data2_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,drop_last=True)
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
        x1 = self.data1_x[index]
        x2 = self.data2_x[index]
        y = [self.data1_y[index], self.data2_y[index]]
        return np.array(x1),np.array(x2), np.array(y)




# 定义模型
class MultitaskModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_tasks,device,batch_size):
        super(MultitaskModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_tasks = num_tasks
        self.device = device
        self.batch_size = batch_size
        self.feature1 = nn.Sequential(

            nn.Linear(1,8),
            nn.ReLU(),
            nn.Linear(8,1),
            nn.ReLU()
        )
        self.feature2 = nn.Sequential(
            nn.Linear(1,8),
            nn.ReLU(),
            nn.Linear(8,1),
            nn.ReLU()
        )
        self.lstm1 = nn.LSTM(1, hidden_size, num_layers, batch_first=True,bidirectional=True)
        self.lstm11 = nn.LSTM(hidden_size*2, hidden_size, num_layers, batch_first=True,bidirectional=True)
        self.lstm2 = nn.LSTM(1, hidden_size, num_layers, batch_first=True,bidirectional=True)
        self.lstm22 = nn.LSTM(hidden_size*2, hidden_size, num_layers, batch_first=True,bidirectional=True)
        self.out1 = nn.Sequential(
            nn.Linear(hidden_size*2,128),
            nn.Linear(128,1)
        )
        self.out2 = nn.Sequential(
            nn.Linear(hidden_size*2,128),
            nn.Linear(128,1)
        )
        self.lstm_all = nn.LSTM(2, hidden_size, num_layers, batch_first=True,bidirectional=True)
        self.lstm_all2 = nn.LSTM(hidden_size*2, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.lstm_all3 = nn.LSTM(hidden_size*2, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_size*2, 128)
        self.fc2 = nn.Linear(128, num_tasks)

    def forward(self, x1,x2):
        #x1 = self.feature1(x1)
        #x2 = self.feature2(x2)
        x = torch.cat((x1,x2),dim=1)

        #处理x1的单独预测
        x1 = x1.unsqueeze(1) 
        
        l1_out, _ = self.lstm1(x1)
        l1_out = l1_out[:, -1, :]
        l1_out,_ = self.lstm11(l1_out.unsqueeze(1))
        l1_out = self.out1(l1_out[:, -1, :])

        #处理x2的单独预测
        x2 = x2.unsqueeze(1) 
        l2_out, _ = self.lstm2(x2)
        l2_out = l2_out[:, -1, :]
        l2_out,_ = self.lstm22(l2_out.unsqueeze(1))
        l2_out = self.out2(l2_out[:, -1, :])

        #共享lstm层
        x = x.unsqueeze(1)
        all_out, _ = self.lstm_all(x)
        all_out = all_out[:, -1, :]
        all_out, _ = self.lstm_all2(all_out.unsqueeze(1))
        all_out = all_out[:, -1, :]
        all_out, _ = self.lstm_all3(all_out.unsqueeze(1))
        all_out = self.fc1(all_out[:, -1, :])
        all_out = self.fc2(all_out)

        # 使用 torch.stack 将两个 tensor 在第0维度进行堆叠
        stacked_tensor = torch.stack([l1_out, l2_out], dim=0)

        # 使用 torch.transpose 将第0维和第1维进行交换
        result_tensor = torch.transpose(stacked_tensor, 0, 1)
        result_tensor = result_tensor.reshape(self.batch_size,2)
        out = 0.6*result_tensor+0.4*all_out
        
        return out
def train(train_loader,batch_size):
    # 超参数
    num_epochs = 20
    learning_rate = 0.01
    # 初始化模型
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device('cpu')
    model = MultitaskModel(input_size=2, hidden_size=64, num_layers=2, num_tasks=2,device=device,batch_size=batch_size)
    #model = nn.DataParallel(model)
    model.to(device)
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 模型训练
    MinTrainLoss = 1e10
    for epoch in range(num_epochs):
        train_loss = 0
        for x1,x2,y in train_loader:
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.to(device)
            y = y.reshape(-1,2)
            outputs= model(x1,x2)
            loss = criterion(outputs,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        if train_loss<MinTrainLoss:
            torch.save(model.state_dict(),"model.pth")
        # 打印训练信息
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(train_loader):.4f}')


def test(data1,data2,batch_size):
    device = torch.device('cpu')
    model = MultitaskModel(input_size=2, hidden_size=64, num_layers=2, num_tasks=2,device = device,batch_size=batch_size)
    model.load_state_dict(torch.load("model.pth"))
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
        for i,(x1,x2,labels) in enumerate(eval_loader):
            x1 = x1.to(device)
            x2 = x2.to(device)
            outputs= model(x1,x2)
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

if __name__ == '__main__':
    batch_size = 16
    data1 = read_data('data\data5.csv')
    data2 = read_data('data\data13.csv')
    train_loader = dataLoad(data1,data2,batch_size)
    train(train_loader,batch_size)
    test(data1,data2,1)
