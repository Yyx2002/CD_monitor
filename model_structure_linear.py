import torch.nn as nn
import torch
from opmatutils import p_real_Linear_module as opt_matrix

# 完成回归任务
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.ln_el1 = torch.nn.Linear(16, 10)# output: (batch, 10)
        self.ln_relu = torch.nn.ReLU()
        self.ln_opt1 = opt_matrix(10, 10, bias=False)# output: (batch, 10)

    def forward(self, x):
        x = self.ln_flat(x)
        x = self.ln_el1(x)
        x = self.ln_relu(x)
        x = self.ln_opt1(x)
        return x