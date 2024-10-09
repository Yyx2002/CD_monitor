from torch.utils.data import Dataset

# 创建数据集
class SignalDataset(Dataset):
    def __init__(self, w, l):
        self.w = w
        self.l = l
    def __len__(self):
        return len(self.w)
    def __getitem__(self, idx):
        return self.w[idx], self.l[idx]