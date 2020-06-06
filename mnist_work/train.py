import torch
from mnist_data import *
from MLP_net import *
from torch.utils.data import DataLoader
from torch import optim
from torch.utils.tensorboard import SummaryWriter

DEVICE = "cuda:0" # 将网络放进cuda 没有cuda就换成cpu


class Train:

    def __init__(self, root):

        self.summaryWriter = SummaryWriter("./logs")

        # 加载训练数据
        self.train_dataset = MNISTDataset(root, True)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=100, shuffle=True)

        # 加载测试数据
        self.test_dataset = MNISTDataset(root, False)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=100, shuffle=True)

        # 创建模型
        self.net = NetV2()
        # self.net.load_state_dict(torch.load("./checkpoint/2.t"))
        self.net.to(DEVICE)
        # 创建优化器
        self.opt = optim.Adam(self.net.parameters())

    # 训练代码
    def __call__(self):
        for epoch in range(100000):
            sum_loss = 0.
            for i, (imgs, tags) in enumerate(self.train_dataloader):
                imgs, tags = imgs.to(DEVICE), tags.to(DEVICE)

                self.net.train()

                y = self.net(imgs)
                loss = torch.mean((tags - y) ** 2)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                sum_loss += loss.cpu().detach().item()

            avg_loss = sum_loss / len(self.train_dataloader)

            sum_score = 0.
            test_sum_loss = 0.
            for i, (imgs, tags) in enumerate(self.test_dataloader):
                imgs, tags = imgs.to(DEVICE), tags.to(DEVICE)

                self.net.eval()

                test_y = self.net(imgs)
                test_loss = torch.mean((tags - test_y) ** 2)
                test_sum_loss += test_loss.cpu().detach().item()

                predict_tags = torch.argmax(test_y, dim=1)
                label_tags = torch.argmax(tags, dim=1)
                sum_score += torch.sum(torch.eq(predict_tags, label_tags).float()).cpu().detach().item()

            test_avg_loss = test_sum_loss / len(self.test_dataloader)
            score = sum_score / len(self.test_dataset)

            self.summaryWriter.add_scalars("loss", {"train_loss": avg_loss, "test_loss": test_avg_loss}, epoch)
            self.summaryWriter.add_scalar("score", score, epoch)

            print(epoch, avg_loss, test_avg_loss, score)

            torch.save(self.net.state_dict(), f"./checkpoint/{epoch}.t")


if __name__ == '__main__':
    train = Train("D:/AIstudyCode/data/MNIST/MNIST_IMG")
    train()
