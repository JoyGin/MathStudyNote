import torch
import pandas as pd
import numpy as np
import torch.utils.data as Data
from matplotlib import pyplot as plt
import random
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
from util.data_1_analysis import analysis_1

num_inputs = 1  #特征数个数
batch_size = 30


analysis = analysis_1()
fea_y = analysis.get_site_np[:, 0]
fea_x = analysis.get_site_np[:, 1]
print(fea_x.shape)

features = torch.tensor(fea_x, dtype=torch.float64)
labels = torch.tensor(fea_y, dtype=torch.float64)

# 将训练数据的特征和标签组合
dataset = Data.TensorDataset(features, labels) 
# 随机读取小批量v
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True) 


# 定义模型
class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)

    def forward(self, x):
        x = x.reshape(-1, 1)
        y = self.linear(x)
        return y


net = LinearNet(num_inputs).double()
print(net)
# print(net[0])


# 初始化模型参数
# print('weight:',net[0].weight)
init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)  # 也可以直接修改bias的data: net[0].bias.data.fill_(0)


# 定义损失函数
loss = nn.MSELoss()

# 定义优化算法
optimizer = optim.SGD(net.parameters(), lr=0.00001)
'''
for i in net.parameters():
    print(i)
'''
# 调整学习率
#for param_group in optimizer.param_groups:
#    param_group['lr'] *= 0.1 # 学习率为之前的0.1倍

# 训练模型
num_epochs = 100
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)
        #print(output.shape)
        l = loss(output, y.view(-1, 1))
        # print(l)
        # print(optimizer)
        l.backward()
        optimizer.step()
        optimizer.zero_grad()  # 梯度清零，等价于net.zero_grad()
    print('epoch %d, loss: %f' % (epoch, l.item()))


dense = net.linear
test_data = torch.tensor([22.566142254795], dtype=torch.float64)
test_result = net(test_data)
print('\n预测经度:',test_result)
print('W:',dense.weight)
print('B:',dense.bias)