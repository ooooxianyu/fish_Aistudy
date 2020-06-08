from torch import nn
import torch


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(3,16,7,2,3),
            nn.ReLU(),
            nn.Conv2d(16,32,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,1,1),
            nn.ReLU(),
            nn.Conv2d(128,128,1,1,0),
            )

    def forward(self,img):
        return self.sequential(img) #1 3 60 240 - 1 128 7 30

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.GRU(128*7*30,128,2,batch_first=True) # 2层GRU
        self.output_layer = nn.Linear(128,10)

    def forward(self,img):
        img = img.reshape(-1,128*7*30)
        img = img[:,None,:].repeat(1,4,1) # 1 4 128*7*30
        #print(img.shape)
        h0 = torch.randn(2,img.size(0),128).cuda() # 2层 批次 输出层
        output, hn = self.rnn(img,h0) # 1 4 128
        #print(output.shape,hn.shape)
        outputs = self.output_layer(output)
        return outputs

class CNN2SEQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self,img):
        h = self.encoder(img)
        y = self.decoder(h)
        return y

if __name__ == '__main__':
    #en_net = Encoder()
    #de_net = Decoder()
    #y = en_net(torch.randn(1, 3, 60, 240))
    #h = de_net(y)
    #print(h.shape)

    net = CNN2SEQ()
    y = net(torch.randn(1, 3, 60, 240))
    print(y.shape)