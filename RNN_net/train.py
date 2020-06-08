import torch
from data import *
from cnn2seq import *
from torch.utils.data import DataLoader
from torch import optim


class Trainer:

    def __init__(self):
        train_dataset = MyDataset("D:/AIstudyCode/data/nums_code/train")
        test_dataset = MyDataset("D:/AIstudyCode/data/nums_code/test")
        self.train_data = DataLoader(train_dataset, 200, shuffle=True)
        self.test_data = DataLoader(test_dataset, 200, shuffle=True)

        self.net = CNN2SEQ().cuda()

        self.opt = optim.Adam(self.net.parameters(),lr=0.001)

        self.loss_fn = nn.CrossEntropyLoss()

    def __call__(self):
        for epoch in range(10000):
            train_sum_loss = 0
            for i, (img, tag) in enumerate(self.train_data):
                # print(img.shape) # 200 * 3 * 60 * 240
                # print(tag.shape) # 200 * 4
                # print(tag)
                img = img.cuda()
                tag = tag.cuda()

                output = self.net(img)
                # print(output.shape) # 200 * 4 * 10 (one-hot)

                _output = output.reshape(-1, 10)
                _tag = tag.reshape(-1)

                loss = self.loss_fn(_output, _tag.long())

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                # print(loss)
                train_sum_loss += loss

            train_avgloss = train_sum_loss / len(img)

            test_sum_loss = 0
            sum_score = 0.
            for i, (img, tag) in enumerate(self.test_data):
                # print(len(self.test_data), len(img))
                img = img.cuda()
                tag = tag.cuda()

                output = self.net(img)

                _output = output.reshape(-1, 10)
                _tag = tag.reshape(-1)

                loss = self.loss_fn(_output, _tag.long())

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                # print(loss)
                test_sum_loss += loss

                # test_tags = torch.argmax(tag,dim=1)
                predict_tags = torch.argmax(output, dim=2)
                # print(predict_tags)
                # print(tag.shape)

                c = torch.eq(predict_tags, tag)
                # print(c.shape)

                for j in range(0, len(img)):
                    if False in c[j]:
                        continue
                    else:
                        sum_score += 1
                # sum_score += torch.sum(torch.eq(predict_tags,tag).float()).cpu().detach().item()

            test_avgloss = test_sum_loss / len(img)
            score = sum_score / len(img)
            # print(len(img))
            print("train:", train_avgloss, "test:", test_avgloss)
            print("score:", score)


if __name__ == '__main__':
    train = Trainer()
    train()

