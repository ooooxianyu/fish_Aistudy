from data import *
from MTCNN_net import *
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
import torch

class Train:
    def __init__(self,root,img_size):
        self.img_size = img_size

        self.mydataset = MyDataset(root,img_size)
        self.dataloader = DataLoader(self.mydataset,batch_size=500,shuffle=True)

        if self.img_size == 12:
            self.net = PNet()
            # 损失训练时候的权重分布
            self.box_p = 0.7
            self.p_p = 0.5
        elif self.img_size == 24:
            self.net = RNet()
            self.box_p = 0.7
            self.p_p = 0.5
        elif self.img_size == 48:
            self.net = ONet()
            self.box_p = 0.5
            self.p_p = 1

        self.net.cuda()

        self.opt= optim.Adam(self.net.parameters())

    def __call__(self, epochs):
        for epoch in range(epochs):
            sum_loss = 0
            for i, (img,tag) in enumerate(self.dataloader):
                img = img.cuda()
                tag = tag.cuda()

                predict = self.net(img)
                if self.img_size == 12:
                    predict = predict.reshape(-1,15)

                torch.sigmoid_(predict[:,0])

                c_mask = tag[:,0]<2
                c_predict = predict[c_mask]
                c_tag = tag[c_mask]

                loss_c = torch.mean((c_predict[:,0] - c_tag[:,0])**2)

                off_mask = tag[:,0]>0
                off_predict = predict[off_mask]
                off_tag = tag[off_mask]

                loss_off_box = torch.mean((off_predict[:,1:5] - off_tag[:,1:5])**2)
                loss_off_px = torch.mean((off_predict[:, 5:] - off_tag[:, 5:]) ** 2)

                loss = loss_c + self.box_p * loss_off_box + self.p_p * loss_off_px

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                sum_loss += loss

                print("loss",loss)

            avg_loss = sum_loss / len(self.dataloader)

            print(epoch,avg_loss)

            # 按损失的前几位保存权重
            s_loss = avg_loss.detach().cpu().numpy()
            s_loss = str(s_loss)

            if self.img_size == 12:
                torch.save(self.net.state_dict(), f"12/loss/pnet_{s_loss[2:5]}.pt")
            elif self.img_size == 24:
                torch.save(self.net.state_dict(), f"24/loss/rnet_{s_loss[2:5]}.pt")
            elif self.img_size == 48:
                torch.save(self.net.state_dict(), f"48/loss/onet_{s_loss[2:5]}.pt")

if __name__ == '__main__':
    train = Train(r"D:/AIstudyCode/MTCNN_new", 12)
    train(10000)