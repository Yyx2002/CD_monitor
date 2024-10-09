import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 此文件对神经网络训练的过程进行模块化
# 可直接通过调用类实现对数据集的下载及分装，并对模型的训练和测试

class DLModule(nn.Module):
    def __init__(self, model, loss_fn, optim, train_set, test_set, train_batch, test_batch, val_set=None, val_batch=None):
        super(DLModule, self).__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optim = optim
        self.writer = SummaryWriter("logs")
        self.train_set = train_set
        self.test_set = test_set
        self.val_set = val_set
        self.train_batch = train_batch
        self.test_batch = test_batch
        self.val_batch = val_batch
        self.train_loader = DataLoader(self.train_set, batch_size=self.train_batch, shuffle=True)
        self.val_loader = DataLoader(self.val_set, batch_size=self.val_batch, shuffle=False)
        self.test_loader = DataLoader(self.test_set, batch_size=self.test_batch, shuffle=False)
        self.trainset_size = len(self.train_set)
        self.testset_size = len(self.test_set)
        self.epoch = 0

    def single_mode(self, input_tensor):
        self.model.eval()
        prediction = self.model(input_tensor)
        return prediction

    # 训练模式
    def train_mode(self):
        self.model.train()
        total_train_loss = 0
        total_train_step = 0
        for input, target in self.train_loader:
            predict = self.model(input)
            loss = self.loss_fn(predict, target)
            total_train_loss += loss.item() * self.train_batch
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            del loss
            total_train_step += 1
            print(f"Train steps : {total_train_step}",  f"Loss : {loss.item()}")
        self.writer.add_scalar("train_loss", total_train_loss, self.epoch)
        if self.val_set != None:
            # 每隔一定次数进行验证
            self.val_mode()

    def val_mode(self):
        if (self.epoch + 1) % 2 == 0:
            self.model.eval()
            val_loss = 0.0
            for input, target in self.val_loader:
                input, target = input.to(torch.float32), target.to(torch.float32)
                predict = self.model(input)
                loss = self.optim(predict, target.unsqueeze(1))
                val_loss += loss.item()
            print(f"Validation Loss: {val_loss/len(self.val_loader):.4f}")
            self.writer.add_scalar("Val_Loss:", loss, self.epoch)

    # 测试模式
    def test_mode(self):
        self.model.eval()
        self.test_results = []
        self.true_labels = []
        total_test_loss = 0
        total_test_accurate_num = 0
        for img, label in self.test_loader:
            # with torch.no_grad():
            predict = self.model(img)
            loss = self.loss_fn(predict, label)
            self.test_results.append(torch.argmax(predict, 1))
            self.true_labels.append(label)
            total_test_loss += loss.item() * self.test_batch
            accurate_num = (predict.argmax(1) == label).sum().item()
            total_test_accurate_num += accurate_num
        self.accuracy = total_test_accurate_num/self.testset_size
        print(f"Test loss: {total_test_loss:.6f}")
        print(f"Test accuracy : {total_test_accurate_num / self.testset_size}")
        self.writer.add_scalar("test_loss", total_test_loss, self.epoch)
        self.writer.add_scalar("test_accuracy", self.accuracy, self.epoch)