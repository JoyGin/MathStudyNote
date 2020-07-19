from util.dataset import TxtDataset
from util.preprocess import preprocess
from util.model import TxtModel
import torch
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
device =torch.device("cuda" if torch.cuda.is_available() else "cpu") 
import os

n_dim = 20000
# 输出的类别为 2
n_categories = 2
# 学习率过大会导致 loss 震荡
learning_rate = 0.001

# 迭代次数
epochs = 50
# 每次迭代同时加载的个数
batch_size = 100

best_accuracy = 0.0


# 损失函数
criterion = nn.CrossEntropyLoss()

model = TxtModel(n_dim, 2).double()
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

#exp_lr_scheduler = lr_scheduler.StepLR(optimizer,step_size = 25, gamma= 0.1)

pre = preprocess()
train_vec_data, test_vec_data, train_label , test_label = pre.get_data()

train_dataset = TxtDataset(train_vec_data, train_label)
test_dataset = TxtDataset(test_vec_data,test_label)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset,batch_size=batch_size, shuffle=False)

for epoch in range(epochs):
    
    model.train()
    loss_total = 0
    st = time.time()
    # train_dataloader 加载数据集
    
    for i, (data, label) in enumerate(train_dataloader):

        data = data.to(device)
        label = label.to(device)

        output = model(data)
        # 计算损失
        loss = criterion(output, label)
        optimizer.zero_grad()
        #exp_lr_scheduler.step()
        # 反向传播
        loss.backward()
        optimizer.step()
        loss_total += loss.item()

    # 输出损失、训练时间等
    print('epoch {}/{}:'.format(epoch, epochs))
    print('training loss: {}, time resumed {}s'.format(
        loss_total/len(train_dataset), time.time()-st))

    model.eval()

    loss_total = 0
    st = time.time()

    correct = 0
    for data, label in test_dataloader:
        data = data.to(device)
        label = label.to(device)

        output = model(data)
        loss = criterion(output, label)
        loss_total += loss.item()

        _, predicted = torch.max(output.data, 1)
        correct += (predicted == label).sum().item()

    # 如果准确度取得最高，则保存准确度最高的模型
    if correct/len(test_dataset) > best_accuracy:
        best_accuracy = correct/len(test_dataset)
        torch.save(model.state_dict(), r"checkpoints/moviePointEpoch%di%d.pth" % (epoch, i))

    print('testing loss: {}, time resumed {}s, accuracy: {}'.format(
        loss_total/len(test_dataset), time.time()-st, correct/len(test_dataset)))