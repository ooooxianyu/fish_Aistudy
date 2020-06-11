import torch
from torch import nn

class DNet(nn.Module):
    def __init__(self):
        super(DNet, self).__init__()

        self.sequential = nn.Sequential(
            nn.Conv2d(3, 64, 5, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self,img):
        h = self.sequential(img)
        return h.reshape(-1)

class GNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.ConvTranspose2d(128, 512, kernel_size=7, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(64, 3, kernel_size=7, stride=3, padding=0, bias=False),
            nn.Tanh()
        )

    def forward(self, noise):
        return self.sequential(noise)

class DCGAN(nn.Module):

    def __init__(self):
        super().__init__()
        self.dnet = DNet()
        self.gnet = GNet()

        self.loss_fn = nn.BCEWithLogitsLoss() #二值交叉熵

    def forward(self,noise):
        return self.gnet(noise)

    def get_D_los(self, noise_d, real_img):
        real_y = self.dnet(real_img)

        g_img = self.gnet(noise_d)
        fake_y = self.dnet(g_img)

        real_tag = torch.ones(real_img.size(0)).cuda()
        fake_tag = torch.zeros(noise_d.size(0)).cuda()

        loss_real = self.loss_fn(real_y, real_tag)
        loss_fake = self.loss_fn(fake_y, fake_tag)

        loss_d = loss_fake + loss_real
        return loss_d

    def get_G_loss(self, noise_g):
        _g_img = self.gnet(noise_g)
        _real_y = self.dnet(_g_img)

        _real_tag = torch.ones(_g_img.size(0)).cuda()

        loss_g = self.loss_fn(_real_y, _real_tag)
        return loss_g

if __name__ == '__main__':
    # dnet = DNet()
    # x = torch.randn(1,3,256,256)
    # y = dnet(x)
    # print(y.shape)

    # gnet = GNet()
    # x = torch.randn(2,128,1,1)
    # y = gnet(x)
    # print(y.shape)

    gan = DCGAN().cuda()
    x = torch.randn(2, 128, 1, 1).cuda()
    r = torch.randn(4, 3, 256, 256).cuda()
    loss_d = gan.get_D_loss(x, r)

    # loss_g = gan.get_G_loss(x)
    # print(loss_d,loss_g)