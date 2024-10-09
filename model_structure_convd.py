import torch.nn as nn
import torch
from opmatutils import p_real_Linear_module as opt_matrix

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # input: (batch, 1, 28, 28)
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 3, 1, 1),# output: (batch, 16, 28, 28)
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),# output: (batch, 16, 14, 14)
            
            torch.nn.Conv2d(16, 32, 3, 1, 1),# output: (batch, 32, 14, 14)
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),# output: (batch, 32, 7, 7)
        )

        self.ln2 = torch.nn.Flatten()# output: (batch, 32 * 7 * 7)
        self.ln3 = torch.nn.Linear(7 * 7 * 32, 10)# output: (batch, 64)
        self.ln4 = torch.nn.ReLU()
        self.ln5 = opt_matrix(10, 10, bias=False)# output: (batch, 10)
        self.ln6 = opt_matrix(10, 10, bias=False)# output: (batch, 10)

    def forward(self, x):
        x = self.features(x)
        x = self.ln2(x)
        x = self.ln3(x)
        x = self.ln4(x)
        x = self.ln5(x)
        x = x * x.shape[-1]
        x = self.ln6(x)
        x = x * x.shape[-1]

        return x