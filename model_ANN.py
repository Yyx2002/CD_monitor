import torch
import torch.nn as nn

# 定义模型
class ANNModel(nn.Module):
    def __init__(self):
        super(ANNModel, self).__init__()
        self.fc_e1 = nn.Linear(80, 160)
        self.fc_e2 = nn.Linear(160, 160)
        self.fc_e3 = nn.Linear(160, 40)
        self.fc_e4 = nn.Linear(40, 40)

    def forward(self, x):
        x = torch.relu(self.fc_e1(x))
        x = torch.relu(self.fc_e2(x))
        x = torch.relu(self.fc_e3(x))
        x = self.fc_e4(x) # 输出层不使用激活函数，因为是回归任务
        return x

