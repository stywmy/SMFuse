# from typing import Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class U2Net(nn.Module):
    def __init__(self):
        super(U2Net, self).__init__()
        self.residual_layer1 = self.make_layer(Conv_ReLU_Block, 7)
        self.residual_layer2 = self.make_layer(Conv_ReLU_Block, 7)
        self.residual_layer3 = self.make_layer(Conv_ReLU_Block, 7)
        self.residual_layer4 = self.make_layer(Conv_ReLU_Block, 7)
        self.residual_layer5 = self.make_layer(Conv_ReLU_Block, 7)
        self.residual_layer6 = self.make_layer(Conv_ReLU_Block, 7)
        self.residual_layer7 = self.make_layer(Conv_ReLU_Block, 7)
        #print(self.residual_layer[5])
        self.input1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=7, stride=1, padding=3, bias=False)
        self.input2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False)
        self.input3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)

        self.output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.output1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
        self.output2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.output3 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)

        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():# 依次来返回模型中的各个层
            if isinstance(m, nn.Conv2d):        # 判断是否是相同的实例对象的。
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def layer1(self, x):
        out = self.relu(self.input1(x))
        return out

    def layer2(self, x):
        out = self.relu(self.input2(x))
        return out

    def layer3(self, x):
        out = self.relu(self.input3(x))
        return out

    def forward(self, x):
        out = self.residual_layer1(x)
        out = self.residual_layer2(out)
        out = self.residual_layer3(out)
        out = self.residual_layer4(out)
        out = self.residual_layer5(out)
        out = self.residual_layer6(out)
        out = self.residual_layer7(out)
        out = self.output1(out)
        out = self.output2(out)
        out = self.output3(out)
        return out

    def fusion_channel_sf(self, f1, f2, kernel_radius = 5):
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        b, c, h, w = f1.shape
        r_shift_kernel = torch.FloatTensor([[0, 0, 0], [1, 0, 0], [0, 0, 0]]).to(device).reshape((1, 1, 3, 3)).repeat(c,
                                                                                                                      1,
                                                                                                                      1,
                                                                                                                      1)
        b_shift_kernel = torch.FloatTensor([[0, 1, 0], [0, 0, 0], [0, 0, 0]]).to(device).reshape((1, 1, 3, 3)).repeat(c,
                                                                                                                      1,
                                                                                                                      1,
                                                                                                                      1)
        f1_r_shift = F.conv2d(f1, r_shift_kernel, padding=1, groups=c)
        f1_b_shift = F.conv2d(f1, b_shift_kernel, padding=1, groups=c)
        f2_r_shift = F.conv2d(f2, r_shift_kernel, padding=1, groups=c)
        f2_b_shift = F.conv2d(f2, b_shift_kernel, padding=1, groups=c)
        f1_grad = torch.pow((f1_r_shift - f1), 2) + torch.pow((f1_b_shift - f1), 2) + 0.001
        f2_grad = torch.pow((f2_r_shift - f2), 2) + torch.pow((f2_b_shift - f2), 2) + 0.001
        fenmu = f1_grad + f2_grad
        f1_weight = f1_grad / fenmu
        f2_weight = f2_grad / fenmu
        f11 = torch.mul(f1, f1_weight)
        f22 = torch.mul(f2, f2_weight)
        dm_tensor1 = f11 + f22
        return dm_tensor1