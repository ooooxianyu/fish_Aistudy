import torch
import torchaudio
from torch.nn import functional as F
from torch import nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = torch.nn.Sequential(
            torch.nn.Conv2d(1, 4, (1, 3), (1, 2), (0, 1)),
            torch.nn.BatchNorm2d(4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(4, 4, (1, 3), (1, 2), (0, 1)),
            torch.nn.BatchNorm2d(4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(4, 4, (1, 3), (1, 2), (0, 1)),
            torch.nn.BatchNorm2d(4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(4, 8, 7, 2, 1),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 16, 5, 1, 0),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            #torch.nn.Conv2d(8, 1, (10, 1)),
            #torch.nn.ReLU(),
        )
        self.lstm = nn.LSTM(10 * 16, 128, 2, batch_first=True, bidirectional=False)
        self.output_layer = nn.Linear(128,35)
        self.apply(weight_init)

    def forward(self, x):
        y = self.seq(x)
        _n, _c, _h, _w = y.shape
        _x = y.permute(0, 2, 3, 1)
        _x = _x.reshape(_n, _h, _w * _c)

        h0 = torch.zeros(2 * 1, _n, 128).cuda()
        c0 = torch.zeros(2 * 1, _n, 128).cuda()
        hsn, (hn, cn) = self.lstm(_x, (h0, c0))
        out = self.output_layer(hsn[:, -1, :])

        return out

def weight_init(m):
    if (isinstance(m, nn.Conv2d)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif (isinstance(m, nn.Linear)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

if __name__ == '__main__':
    net = Net().cuda()
    y = net(torch.randn(1,1,32,256).cuda())
    print(y.shape)

