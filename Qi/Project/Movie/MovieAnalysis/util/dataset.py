from preprocess import preprocess
import torch
from torch.utils.data import Dataset, DataLoader

class TxtDataset(Dataset):


    def __init__(self, vec_datas, labels):
        self.vec_datas = vec_datas
        self.labels = labels

    
    def __getitem__(self, index):
        # DataLoader 会根据 index 获取数据
        # toarray() 是因为 VectData 是一个稀疏矩阵，如果直接使用 VectData.toarray() 占用内存太大，勿尝试
        return self.vec_datas[index].toarray(), self.labels[index]-1
    
    def __len__(self):
        return len(self.labels)


if __name__ == "__main__":

    pre = preprocess()

    train_vec_data, test_vec_data, train_label , test_label = pre.get_data()
    train_dataset = TxtDataset(train_vec_data, train_label)
    train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True)

    for data, label in train_dataloader:
        print(data,label)


    
