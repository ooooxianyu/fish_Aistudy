import dataset
from model import yoloV3_net
import torch
from torch.utils.data import DataLoader
import os

# 损失
def loss_fn(output, target, alpha):

    # torch.nn.BCELoss() 必须得用sigmoid激活
    conf_loss_fn = torch.nn.BCEWithLogitsLoss() # 二分类交叉熵 -> 置信度 自带sigmoid+BCE
    crood_loss_fn = torch.nn.MSELoss() # 平方差 -> box位置
    cls_loss_fn = torch.nn.CrossEntropyLoss() # 交叉熵 -> 类别
    # NLLLOSS log+softmax

    # N C H W -> N H W C
    output = output.permute(0, 2, 3, 1)
    # N C H W -> N H W 3 15
    output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)
    output = output.cpu().double()
    mask_obj = target[..., 0] > 0
    output_obj = output[mask_obj]
    target_obj = target[mask_obj]

    loss_obj_conf = conf_loss_fn(output_obj[:,0], target_obj[:,0]) # 置信度损失
    loss_obj_crood = crood_loss_fn(output_obj[:,1:5],target_obj[:,1:5]) # box损失
    # 交叉熵的损失函数 第一个预测出来的标签，且是onehot 第二个参数为真实标签
    label_tags = torch.argmax(target_obj[:,5:], dim=1)
    #print(output_obj[:, 5:].shape, label_tags.shape)
    loss_obj_cls = cls_loss_fn(output_obj[:, 5:], label_tags) # 类别损失

    loss_obj = loss_obj_cls + loss_obj_conf + loss_obj_crood # 总损失

    mask_noobj = target[..., 0] == 0 # anchor中iou为0的数据进行训练 即置信度为0 负样本 只需要训练与真实标签的置信度
    output_noobj = output[mask_noobj]
    target_noobj = target[mask_noobj]
    loss_noobj = conf_loss_fn(output_noobj[:,0], target_noobj[:, 0])
    # 通过权重进行相加 （权重的比例可根据数据集 正负样本比例来设置）
    loss = alpha * loss_obj + (1 - alpha) * loss_noobj

    return loss

if __name__ == '__main__':

    save_path = "data/checkpoints/myyolo.pt" #权重保存的位置
    myDataset = dataset.MyDataset()
    train_loader = DataLoader(myDataset, batch_size=2, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #是否有cuda
    net = yoloV3_net().to(device)

    # if os.path.exists(save_path):
    #     net.load_state_dict(torch.load(save_path)) #加载权重
    # else:
    #     print("NO Param!")

    net.train()
    opt = torch.optim.Adam(net.parameters())

    epoch = 0
    while(True):
        for target_13, target_26, target_52, img_data in train_loader:
            img_data = img_data.to(device)
            output_13, output_26, output_52 = net(img_data)
            #print(output_13.shape, output_26.shape, output_52.shape)
            #print(target_13.shape, target_26.shape, target_52.shape)
            #exit()
            loss_13 = loss_fn(output_13, target_13, 0.4)
            loss_26 = loss_fn(output_26, target_26, 0.4)
            loss_52 = loss_fn(output_52, target_52, 0.4)
            loss = loss_13 + loss_26 + loss_52

            opt.zero_grad()
            loss.backward()
            opt.step()
            if epoch % 10 == 0:
                torch.save(net.state_dict(),save_path)
                print('save{}'.format(epoch))
            print(loss.item())
        epoch += 1