from data import *
from gan_net import *

from torch.utils.data import DataLoader
from torch import optim
from torchvision import utils

class Trainer:

    def __init__(self, root):
        facemydata = FaceMyData()
        self.train_dataloader = DataLoader(facemydata, 200, True)

        self.net = DCGAN().cuda()

        self.d_opt = optim.Adam(self.net.dnet.parameters(), 0.0002, betas=(0.5, 0.9))
        self.g_opt = optim.Adam(self.net.gnet.parameters(), 0.0002, betas=(0.5, 0.9))

    def __call__(self):
        # self.net.dnet.load_state_dict(torch.load("params_face/d_net.pth"))
        # self.net.gnet.load_state_dict(torch.load("params_face/g_net.pth"))
        for epoch in range(10000):
            for i, img in enumerate(self.train_dataloader):
                real_img = img.cuda()

                noise_d = torch.normal(0, 0.1, (100, 128, 1, 1)).cuda()
                noise_g = torch.normal(0, 0.1, (100, 128, 1, 1)).cuda()

                loss_d = self.net.get_D_los(noise_d, real_img)
                self.d_opt.zero_grad()
                loss_d.backward()
                self.d_opt.step()

                loss_g = self.net.get_G_loss()
                self.g_opt.zero_grad()
                loss_g.backward()
                self.g_opt.step()

                print(loss_d.cpu().detach().item(), loss_g.cpu().detach().item())

            print("done")
            torch.save(self.net.dnet.state_dict(), "params_face/d_net.pth")
            torch.save(self.net.gnet.state_dict(), "params_face/g_net.pth")

            noise = torch.normal(0, 0.1, (8, 128, 1, 1)).cuda()
            y = self.net(noise)
            utils.save_image(y, f"img_face/{epoch}.jpg", normalize=True, range=(-1, 1))

if __name__ == '__main__':
    train = Trainer(r"D:\AIstudyCode\data\seeprettyface_chs_wanghong\data")
    train()