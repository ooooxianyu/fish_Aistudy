from data import *
from cnn2_lstm_net import *
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
import torch

from torch.utils.tensorboard import SummaryWriter

class Train:
    def __init__(self, root):
        super().__init__()
        self.mydataset = MyDataset(root)
        self.dataloader = DataLoader(self.mydataset, batch_size=500, shuffle=True)

        self.net = Net().cuda()
        self.opt = optim.Adam(self.net.parameters(), lr=0.001)

    def __call__(self, epochs):
        loss_fn_1 = nn.CrossEntropyLoss()
        print(self.net)

        summaryWriter = SummaryWriter("logs")

        for epoch in range(epochs):
            sum_loss = 0.
            sum_score = 0.
            for i, (audio_data, tag) in enumerate(self.dataloader):
                audio_data = audio_data.cuda()
                tag = tag.cuda()

                output = self.net(audio_data)
                output = output.reshape(-1, 35)

                tag = tag.reshape(-1)

                loss = loss_fn_1(output, tag.long())

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                sum_loss += loss
                print(loss)

                predict_tags = torch.argmax(output, dim=1)
                sum_score += torch.sum(torch.eq(predict_tags, tag).float()).cpu().detach().item()

            avg_loss= sum_loss / len(self.dataloader)
            score = sum_score / len(self.mydataset)
            print("epoch:", epoch, "avg_loss:", avg_loss, "score:", score)

            summaryWriter.add_scalars("loss", {"train_loss": avg_loss}, i)
            summaryWriter.add_scalars("score", {"score": score}, i)

            layer1_weight = self.net.seq[0].weight
            layer2_weight = self.net.seq[3].weight
            layer3_weight = self.net.seq[6].weight
            layer4_weight = self.net.seq[9].weight
            layer5_weight = self.net.seq[12].weight
            # layer6_weight = self.net.rnn[0].weight
            layer7_weight = self.net.output_layer.weight

            summaryWriter.add_histogram("layer1", layer1_weight, i)
            summaryWriter.add_histogram("layer2", layer2_weight, i)
            summaryWriter.add_histogram("layer3", layer3_weight, i)
            summaryWriter.add_histogram("layer4", layer4_weight, i)
            summaryWriter.add_histogram("layer5", layer5_weight, i)
            # summaryWriter.add_histogram("layer6", layer6_weight, i)
            summaryWriter.add_histogram("layer7", layer7_weight, i)

if __name__ == '__main__':
    train = Train("D:/AIstudyCode/data/audio_data/speech_commands_train_set_v0.02/")
    train(1000)