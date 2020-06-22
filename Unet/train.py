import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from UNet import UNet
from gen_data import makeData
from torchvision.utils import save_image

path = r"D:/AIstudyCode/data/VOCtrainval_11-May-2012/VOCdevkit/VOC2012"
module = r"D:/AIstudyCode/data/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/train/module.pth"
img_save_path = r"D:/AIstudyCode/data/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/train/train_save_img"
epoch = 1

net = UNet().cuda()
optimizer = torch.optim.Adam(net.parameters())
loss_func = nn.BCELoss()

dataloader = DataLoader(makeData(path), batch_size=3, shuffle=True)

if os.path.exists(module):
    net.load_state_dict(torch.load(module))
else:
    print("NO Params!")

if not os.path.exists(img_save_path):
    os.mkdir(img_save_path)

while True:
    for i, (xs_jpg,ys_png) in enumerate(dataloader):
        xs_jpg = xs_jpg.cuda()
        ys_png = ys_png.cuda()
        _xs_jpg = net(xs_jpg)

        loss = loss_func(_xs_jpg, ys_png)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i%50 == 0:
            print('epoch:{},count:{},loss:{}'.format(epoch, i, loss))
            x = xs_jpg[0]
            _x = _xs_jpg[0]
            y = ys_png[0]

            img = torch.stack([x, _x, y], 0)
            # print(img.shape)
            torch.save(net.state_dict(), module)
            print('module is saved !')
            save_image(img.cpu(), os.path.join(img_save_path, '{}.png'.format(i)))
            print("saved successfully!")

    epoch += 1