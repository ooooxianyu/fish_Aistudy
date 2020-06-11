import torch, gym, random
from torch import nn, optim
import numpy as np


class QNet(nn.Sequential):

    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, 2),
            nn.ReLU(),
        )

        self.ouput_layer = nn.Sequential(
            nn.Linear(10 * 7 * 128, 256),
            nn.ReLU(),
            nn.Linear(256, 8)
        )

    def forward(self, x):
        h = self.backbone(x)
        h = h.reshape(-1, 10 * 7 * 128)
        return self.ouput_layer(h)


class Game:

    def __init__(self, exp_pool_size, explore=0.9):
        self.env = gym.make('Phoenix-v0')  # Phoenix-v0 MsPacman-v0

        self.exp_pool = []  # 采集样本数据
        self.exp_pool_size = exp_pool_size

        self.q_net = QNet()

        self.explore = explore

        self.loss_fn = nn.MSELoss()

        self.opt = optim.Adam(self.q_net.parameters())

    def __call__(self):
        is_render = False
        avg = 0
        while True:
            # 数据采样
            state = self.env.reset()  # List
            state = state.transpose(2, 1, 0)
            state = (state / 255.).astype(np.float32)
            # print(state.dtype)
            # print(state)
            R = 0  # 总回报
            while True:
                if is_render: self.env.render()

                if len(self.exp_pool) > self.exp_pool_size:
                    # 经验池满了
                    self.exp_pool.pop(0)  # 把旧经验删除
                    self.explore += 0.00001
                    # 当经验值满了但不代表经验选取的结果是最好的

                    if random.random() >= self.explore:
                        action = self.env.action_space.sample()

                    else:
                        # state.astype(np.float32)
                        _state = torch.tensor(state)  # 用 _ 代表tensor变量
                        # _state = _state.long()
                        # print(_state.dtype)
                        # print(_state)
                        Qs = self.q_net(_state[None, ...])  # Qs 得到当前状态下 各个动作的Q值 NV结构
                        # print("in")
                        action = Qs.argmax(dim=1)[0].item()  # 动作 0 1 ... 9 选最大Q值得索引即为动作

                else:
                    # 经验池没满就随机采用
                    action = self.env.action_space.sample()
                    # action = random.randint(0,9)

                next_state, reward, done, _ = self.env.step(action)
                next_state = next_state.transpose(2, 1, 0)
                next_state = (next_state / 255.).astype(np.float32)
                # print(reward)
                # if not done:
                #     reward += 1

                R += reward  # 回报累加

                self.exp_pool.append([state, reward, action, next_state, done])

                state = next_state
                # print(state.dtype)
                # print(state)

                if done:
                    avg = 0.95 * avg + 0.05 * R
                    print(avg, R, self.env.spec.reward_threshold)
                    if avg > 800:  # 训练到平均能走300分时候 开始显示游戏
                        is_render = True
                    break  # 结束时候跳出循环——进行下一次游戏

            # print(len(self.exp_pool))
            # 训练 （边采样边训练 早起随机采样 后期用网络采样）
            if len(self.exp_pool) >= self.exp_pool_size:
                exps = random.choices(self.exp_pool, k=500)  # 在经验池里随机挑选经验（打乱顺序）

                _state = torch.tensor([exp[0].tolist() for exp in exps])
                _reward = torch.tensor([[exp[1]] for exp in exps])
                _action = torch.tensor([[exp[2]] for exp in exps])
                _next_state = torch.tensor([exp[3].tolist() for exp in exps])
                _done = torch.tensor([[int(exp[4])] for exp in exps])

                # 得到估计值
                _Qs = self.q_net(_state)
                # print(_Qs,_action)
                _Q = torch.gather(_Qs, 1, _action)

                # 目标值
                _next_Qs = self.q_net(_next_state)
                _max_Q = _next_Qs.max(dim=1, keepdim=True)[0]
                _target_Q = _reward + (1 - _done) * 0.9 * _max_Q  # Q函数

                loss = self.loss_fn(_Q, _target_Q.detach())

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()


if __name__ == '__main__':
    game = Game(20000)
    game()

# if __name__ == '__main__':
#    #pnet = PNet()
#    #y = pnet(torch.randn(1,3,12,12))
#    #y = y.reshape(1,15)
#    #print(y.shape)

#    #rnet = RNet()
#    #y = rnet(torch.randn(1,3,24,24))
#    #print(y.shape)

#    net = QNet()
#    y = net(torch.randn(1,3,210,160))
#    print(y.shape)