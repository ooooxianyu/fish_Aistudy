import torch
import torch.nn.functional as F
from torch import nn

# 上采样层：临近插值
class UpsampleLayer(nn.Module):
    def __init__(self):
        super(UpsampleLayer,self).__init__()

    def forward(self,x):
        # mode 为上/下采样的算法
        return F.interpolate(x,scale_factor=2,mode='nearest')

# 卷积层 DBL： conv BN LeakyRelU
class ConvolutionalLayer(nn.Module):
    def __init__(self,in_ch,out_ch,kernel_s,stride,padding,bias=False):
        super(ConvolutionalLayer,self).__init__()

        self.sub_module = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,kernel_s,stride,padding,bias = bias),
            nn.BatchNorm2d(out_ch),
            #改成frn
            nn.LeakyReLU(0.1)
        )
    def forward(self,x):
        return self.sub_module(x)

# 残差网络
class ResidualLayer(nn.Module):
    def __init__(self,in_ch):
        super(ResidualLayer,self).__init__()
        self.sub_module = nn.Sequential(
            ConvolutionalLayer(in_ch,in_ch//2,1,1,0),
            ConvolutionalLayer(in_ch//2,in_ch,3,1,1),
        )

    def forward(self,x):
        return x + self.sub_module(x)

# 定义下采样层
class DownsamplingLayer(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(DownsamplingLayer,self).__init__()
        self.sub_module = nn.Sequential(
            ConvolutionalLayer(in_ch,out_ch,3,2,1)
        )
    def forward(self,x):
        return self.sub_module(x)

#定义卷积快
class ConvolutionalSet(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(ConvolutionalSet,self).__init__()

        self.sub_module = nn.Sequential(
            # 瓶颈结构
            ConvolutionalLayer(in_ch, out_ch, 1, 1, 0),
            ConvolutionalLayer(out_ch, in_ch, 3, 1, 1),

            ConvolutionalLayer(in_ch, out_ch, 1, 1, 0),
            ConvolutionalLayer(out_ch, in_ch, 3, 1, 1),

            ConvolutionalLayer(in_ch, out_ch, 1, 1, 0),
        )

    def forward(self, x):
        return self.sub_module(x)

class yoloV3_net(nn.Module):
    def __init__(self):
        super(yoloV3_net,self).__init__()

        self.trunk_52 = nn.Sequential(
            ConvolutionalLayer(3,32,3,1,1),
            DownsamplingLayer(32,64),
            ResidualLayer(64),
            DownsamplingLayer(64,128),
            ResidualLayer(128),
            ResidualLayer(128),
            DownsamplingLayer(128,256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
        )

        self.trunk_26 = nn.Sequential(
            DownsamplingLayer(256,512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
        )

        self.trunk_13 = nn.Sequential(
            DownsamplingLayer(512,1024),
            ResidualLayer(1024),
            ResidualLayer(1024),
            ResidualLayer(1024),
            ResidualLayer(1024),
        )

        self.convset_13 = nn.Sequential(
            ConvolutionalSet(1024,512)
        )
        self.detetion_13 = nn.Sequential(
            ConvolutionalLayer(512,1024,3,1,1),
            nn.Conv2d(1024,3*(5+5),1,1,0)
        )

        self.up_26 = nn.Sequential(
            ConvolutionalLayer(512,256,1,1,0),
            UpsampleLayer()
        )
        self.convset_26 = nn.Sequential(
            ConvolutionalSet(768,256)
        )
        self.detetion_26 = nn.Sequential(
            ConvolutionalLayer(256, 512, 3, 1, 1),
            nn.Conv2d(512, 3*(5+5), 1, 1, 0)
        )

        self.up_52 = nn.Sequential(
            ConvolutionalLayer(256, 128, 1, 1, 0),
            UpsampleLayer()
        )
        self.convset_52 = nn.Sequential(
            ConvolutionalSet(384, 128)
        )
        self.detetion_52 = nn.Sequential(
            ConvolutionalLayer(128, 256, 3, 1, 1),
            nn.Conv2d(256, 3*(5+5), 1, 1, 0)
        )

    def forward(self,x):
        # 卷积层得到不同尺寸的特征图
        h_52 = self.trunk_52(x)
        h_26 = self.trunk_26(h_52)
        h_13 = self.trunk_13(h_26)
        #print(h_52.shape)
        #print(h_26.shape)
        #print(h_13.shape)
        #exit()
        # 13
        convset_out_13 = self.convset_13(h_13)
        detetion_out_13 = self.detetion_13(convset_out_13)
        up_13to26 = self.up_26(convset_out_13)
        # 26
        route_out_13and26 = torch.cat((up_13to26,h_26),dim=1)
        convset_out_26 = self.convset_26(route_out_13and26)
        detetion_out_26 = self.detetion_26(convset_out_26)
        up_26to52 = self.up_52(convset_out_26)
        # 52
        route_out_26and52 = torch.cat((up_26to52, h_52), dim=1)
        convset_out_52 = self.convset_52(route_out_26and52)
        detetion_out_52 = self.detetion_52(convset_out_52)
        return detetion_out_13,detetion_out_26,detetion_out_52

if __name__ == '__main__':
    trunk = yoloV3_net()
    # x = torch.Tensor([2,3,416,416])
    x = torch.randn([2,3,416,416],dtype=torch.float32)
    # 测试网络
    y_13, y_26, y_52 = trunk(x)
    print(y_13.shape)
    print(y_26.shape)
    print(y_52.shape)
    print(y_13.view(-1, 3, 15, 13, 13).shape)
