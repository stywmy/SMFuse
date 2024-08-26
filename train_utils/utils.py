import argparse
import os
import torchvision.models as models
import torch.nn as nn
import torch
# from model.unet_model import *

class PerceptualLoss(nn.Module):
    def __init__(self, is_cuda):
        super(PerceptualLoss, self).__init__()
        #print('loading resnet101...')
        net = models.resnet101(num_classes=2)
        #print(net.conv1)
        net.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        pretrained_dict = torch.load("./resnet101.pth")
        model_dict = net.state_dict()
        # 重新制作预训练的权重，主要是减去参数不匹配的层，楼主这边层名为“fc”
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and 'fc' not in k)}
        # 更新权重
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)

        # # model = model_dict[opt.model](num_classes=10)
        # # model_path = "./resnet101.pth"
        # # pre_weights = torch.load(model_path)['model']
        # # model.load_state_dict(pre_weights, strict=False)
        # # self.loss_network.cuda()
        # net = models.resnet101()
        # net_params  = torch.load("./resnet101.pth")
        # net.load_state_dict(net_params)
        # ckpt.pop("fc.bias")
        # ckpt.pop("fc.weight")
        # ckpt = ckpt.load_state_dict(ckpt, strict=False)
        # self.loss_network = ckpt
        # self.loss_network.eval()
        #self.loss_network = models.resnet101(pretrained=True, num_classes=2)
        #model = models.resnet101(pretrained=True, num_classes=2)
        # net.pop("fc.bias")
        # net.pop("fc.weight")
        self.loss_network = net
        # Turning off gradient calculations for efficiency
        for param in self.loss_network.parameters():
            param.requires_grad = False
        if is_cuda:
            self.loss_network.cuda()
        #print("done ...")

    def mse_loss(self, input, target):
        return torch.sum((input - target) ** 2) / input.data.nelement()
    def forward(self, output, label):
        self.perceptualLoss = self.mse_loss(self.loss_network(output),self.loss_network(label))
        return self.perceptualLoss