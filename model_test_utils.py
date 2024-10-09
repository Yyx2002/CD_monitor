import torch
import photontorch as pt
import torchvision
import torch.nn as nn
import numpy as np

# 定义相位随机变化函数
def randphase(model, coe):
    for name, para in model.named_parameters():
        if "mzi" in name:
            stillattr = model
            for a in name.split("."):
                stillattr = getattr(stillattr, a)
            stillattr.data = stillattr.data + coe * np.random.randn()
    return model

class PrecisionLimit(nn.Module):
    def __init__(self, valid_number):
        super(PrecisionLimit, self).__init__()
        self.vn = valid_number

    # 自定义钩子函数，限制输出精度
    def modify_output_precision(self, module, input, output):
        # 假设你想限制为3位有效数字
        return torch.round(output * self.vn) / self.vn

    def creat_hook(self, layer):
        hook_handle = layer.register_forward_hook(self.modify_output_precision)