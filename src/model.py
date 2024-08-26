# # --------------------------------------训练结构图--------------------------------------
# from typing import Union, List
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class ConvBNReLU(nn.Module):
#     def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1):
#         super().__init__()
#
#         padding = kernel_size // 2 if dilation == 1 else dilation
#         self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation, bias=False)
#         self.bn = nn.BatchNorm2d(out_ch)
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.relu(self.bn(self.conv(x)))
#
#
# class DownConvBNReLU(ConvBNReLU):
#     def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, flag: bool = True):
#         super().__init__(in_ch, out_ch, kernel_size, dilation)
#         self.down_flag = flag
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         if self.down_flag:
#             x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)
#         return self.relu(self.bn(self.conv(x)))
#
# class UpConvBNReLU(ConvBNReLU):
#     def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, flag: bool = True):
#         super().__init__(in_ch, out_ch, kernel_size, dilation)
#         self.up_flag = flag
#
#     def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
#         if self.up_flag:
#             x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=False)
#         return self.relu(self.bn(self.conv(torch.cat([x1, x2], dim=1))))
#
#
# class RSU(nn.Module):
#     def __init__(self, height: int, in_ch: int, mid_ch: int, out_ch: int):
#         super().__init__()
#
#         assert height >= 2
#         self.conv_in = ConvBNReLU(in_ch, out_ch)
#
#         encode_list = [DownConvBNReLU(out_ch, mid_ch, flag=False)]
#         decode_list = [UpConvBNReLU(mid_ch * 2, mid_ch, flag=False)]
#         for i in range(height - 2):
#             encode_list.append(DownConvBNReLU(mid_ch, mid_ch))
#             decode_list.append(UpConvBNReLU(mid_ch * 2, mid_ch if i < height - 3 else out_ch))
#
#         encode_list.append(ConvBNReLU(mid_ch, mid_ch, dilation=2))
#         self.encode_modules = nn.ModuleList(encode_list)
#         self.decode_modules = nn.ModuleList(decode_list)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x_in = self.conv_in(x)
#
#         x = x_in
#         encode_outputs = []
#         for m in self.encode_modules:
#             x = m(x)
#             encode_outputs.append(x)
#
#         x = encode_outputs.pop()
#         for m in self.decode_modules:
#             x2 = encode_outputs.pop()
#             x = m(x, x2)
#
#         return x + x_in
#
#
# class RSU4F(nn.Module):
#     def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
#         super().__init__()
#         self.conv_in = ConvBNReLU(in_ch, out_ch)
#         self.encode_modules = nn.ModuleList([ConvBNReLU(out_ch, mid_ch),
#                                              ConvBNReLU(mid_ch, mid_ch, dilation=2),
#                                              ConvBNReLU(mid_ch, mid_ch, dilation=4),
#                                              ConvBNReLU(mid_ch, mid_ch, dilation=8)])
#
#         self.decode_modules = nn.ModuleList([ConvBNReLU(mid_ch * 2, mid_ch, dilation=4),
#                                              ConvBNReLU(mid_ch * 2, mid_ch, dilation=2),
#                                              ConvBNReLU(mid_ch * 2, out_ch)])
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x_in = self.conv_in(x)
#
#         x = x_in
#         encode_outputs = []
#         for m in self.encode_modules:
#             x = m(x)
#             encode_outputs.append(x)
#
#         x = encode_outputs.pop()
#         for m in self.decode_modules:
#             x2 = encode_outputs.pop()
#             x = m(torch.cat([x, x2], dim=1))
#
#         return x + x_in
#
#
# class Conv_ReLU_Block(nn.Module):
#     def __init__(self):
#         super(Conv_ReLU_Block, self).__init__()
#         self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         return self.relu(self.conv(x))
#
#
# class U2Net(nn.Module):
#     def __init__(self, cfg: dict, out_ch: int = 1):
#         super().__init__()
#         assert "encode" in cfg
#         assert "decode" in cfg
#         self.encode_num = len(cfg["encode"])
#         self.relu = nn.ReLU(inplace=True)
#
#         encode_list = []
#         side_list = []
#         for c in cfg["encode"]:
#             # c: [height, in_ch, mid_ch, out_ch, RSU4F, side]
#             assert len(c) == 6
#             encode_list.append(RSU(*c[:4]) if c[4] is False else RSU4F(*c[1:4]))
#
#             if c[5] is True:
#                 side_list.append(nn.Conv2d(c[3], out_ch, kernel_size=3, padding=1))
#         self.encode_modules = nn.ModuleList(encode_list)
#
#         decode_list = []
#         for c in cfg["decode"]:
#             # c: [height, in_ch, mid_ch, out_ch, RSU4F, side]
#             assert len(c) == 6
#             decode_list.append(RSU(*c[:4]) if c[4] is False else RSU4F(*c[1:4]))
#
#             if c[5] is True:
#                 side_list.append(nn.Conv2d(c[3], out_ch, kernel_size=3, padding=1))
#         self.decode_modules = nn.ModuleList(decode_list)
#         self.side_modules = nn.ModuleList(side_list)
#         self.out_conv = nn.Conv2d(self.encode_num * out_ch, out_ch, kernel_size=1)
#
#     def encoder1(self, x: torch.Tensor):
#         _, _, h, w = x.shape
#         m = self.encode_modules[0]
#         x = m(x)
#         return x
#
#     def decoder1(self, en1: torch.Tensor):
#         m = self.decode_modules[0]
#         de1 = m(en1)
#         return de1
#
#     def side(self, de1: torch.Tensor):
#         _, _, h, w = de1.shape
#         m = self.side_modules[0]
#         side1 = m(de1)
#         return side1
#
#
#
# def u2net_full(out_ch: int = 1):
#     cfg = {
#         "encode": [[7, 1, 32, 2, False, True]],  # En1
#         "decode": [[7, 1, 32, 64, False, True]]  # De1
#     }
#
#     return U2Net(cfg, out_ch)
#
#
# def u2net_lite(out_ch: int = 1):
#     cfg = {
#         # height, in_ch, mid_ch, out_ch, RSU4F, side
#         "encode": [[7, 3, 16, 64, False, False],  # En1
#                    [6, 64, 16, 64, False, False],  # En2
#                    [5, 64, 16, 64, False, False],  # En3
#                    [4, 64, 16, 64, False, False],  # En4
#                    [4, 64, 16, 64, True, False],  # En5
#                    [4, 64, 16, 64, True, True]],  # En6
#         # height, in_ch, mid_ch, out_ch, RSU4F, side
#         "decode": [[4, 128, 16, 64, True, True],  # De5
#                    [4, 128, 16, 64, False, True],  # De4
#                    [5, 128, 16, 64, False, True],  # De3
#                    [6, 128, 16, 64, False, True],  # De2
#                    [7, 128, 16, 64, False, True]]  # De1
#     }
#
#     return U2Net(cfg, out_ch)
#
#
# def convert_onnx(m, save_path):
#     m.eval()
#     x = torch.rand(1, 2, 288, 288, requires_grad=True)
#
#     # export the model
#     torch.onnx.export(m,  # model being run
#                       x,  # model input (or a tuple for multiple inputs)
#                       save_path,  # where to save the model (can be a file or file-like object)
#                       export_params=True,
#                       opset_version=11)
#
#
# if __name__ == '__main__':
#     # n_m = RSU(height=7, in_ch=3, mid_ch=12, out_ch=3)
#     # convert_onnx(n_m, "RSU7.onnx")
#     #
#     # n_m = RSU4F(in_ch=3, mid_ch=12, out_ch=3)
#     # convert_onnx(n_m, "RSU4F.onnx")
#
#     u2net = u2net_full()
#     convert_onnx(u2net, "u2net_full.onnx")

# #--------------------------------------训练决策图--------------------------------------
from typing import Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.nets_utility import GaussBlur
from src.guided_filter import GuidedFilter


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()

        padding = kernel_size // 2 if dilation == 1 else dilation
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class DownConvBNReLU(ConvBNReLU):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, flag: bool = True):
        super().__init__(in_ch, out_ch, kernel_size, dilation)
        self.down_flag = flag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.down_flag:
            x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)

        return self.relu(self.bn(self.conv(x)))


class UpConvBNReLU(ConvBNReLU):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, flag: bool = True):
        super().__init__(in_ch, out_ch, kernel_size, dilation)
        self.up_flag = flag

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if self.up_flag:
            x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=False)
        return self.relu(self.bn(self.conv(torch.cat([x1, x2], dim=1))))


class RSU(nn.Module):
    def __init__(self, height: int, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()

        assert height >= 2
        self.conv_in = ConvBNReLU(in_ch, out_ch)

        encode_list = [DownConvBNReLU(out_ch, mid_ch, flag=False)]
        decode_list = [UpConvBNReLU(mid_ch * 2, mid_ch, flag=False)]
        for i in range(height - 2):
            encode_list.append(DownConvBNReLU(mid_ch, mid_ch))
            decode_list.append(UpConvBNReLU(mid_ch * 2, mid_ch if i < height - 3 else out_ch))

        encode_list.append(ConvBNReLU(mid_ch, mid_ch, dilation=2))
        self.encode_modules = nn.ModuleList(encode_list)
        self.decode_modules = nn.ModuleList(decode_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.conv_in(x)

        x = x_in
        encode_outputs = []
        for m in self.encode_modules:
            x = m(x)
            encode_outputs.append(x)

        x = encode_outputs.pop()
        for m in self.decode_modules:
            x2 = encode_outputs.pop()
            x = m(x, x2)

        return x + x_in


class RSU4F(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()
        self.conv_in = ConvBNReLU(in_ch, out_ch)
        self.encode_modules = nn.ModuleList([ConvBNReLU(out_ch, mid_ch),
                                             ConvBNReLU(mid_ch, mid_ch, dilation=2),
                                             ConvBNReLU(mid_ch, mid_ch, dilation=4),
                                             ConvBNReLU(mid_ch, mid_ch, dilation=8)])

        self.decode_modules = nn.ModuleList([ConvBNReLU(mid_ch * 2, mid_ch, dilation=4),
                                             ConvBNReLU(mid_ch * 2, mid_ch, dilation=2),
                                             ConvBNReLU(mid_ch * 2, out_ch)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.conv_in(x)

        x = x_in
        encode_outputs = []
        for m in self.encode_modules:
            x = m(x)
            encode_outputs.append(x)

        x = encode_outputs.pop()
        for m in self.decode_modules:
            x2 = encode_outputs.pop()
            x = m(torch.cat([x, x2], dim=1))

        return x + x_in

class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class U2Net(nn.Module):
    def __init__(self, cfg: dict, out_ch: int = 1):
        super().__init__()
        assert "encode" in cfg
        assert "decode" in cfg
        self.encode_num = len(cfg["encode"])
        self.relu = nn.ReLU(inplace=True)
        self.gaussian = GaussBlur(8, 4)
        self.guided_filter = GuidedFilter(3, 0.1)
        # self.residual_layer1 = self.make_layer(Conv_ReLU_Block, 7)
        # self.residual_layer2 = self.make_layer(Conv_ReLU_Block, 7)
        # self.residual_layer3 = self.make_layer(Conv_ReLU_Block, 7)
        # self.residual_layer4 = self.make_layer(Conv_ReLU_Block, 7)
        # self.residual_layer5 = self.make_layer(Conv_ReLU_Block, 7)
        # self.residual_layer6 = self.make_layer(Conv_ReLU_Block, 7)
        # self.residual_layer7 = self.make_layer(Conv_ReLU_Block, 7)
        # # print(self.residual_layer[5])
        # self.input1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=7, stride=1, padding=3, bias=False)
        # self.input2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False)
        # self.input3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        #
        # self.output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        # self.output1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
        # self.output2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        # self.output3 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)

        encode_list = []
        side_list = []
        for c in cfg["encode"]:
            # c: [height, in_ch, mid_ch, out_ch, RSU4F, side]
            assert len(c) == 6
            encode_list.append(RSU(*c[:4]) if c[4] is False else RSU4F(*c[1:4]))

            if c[5] is True:
                side_list.append(nn.Conv2d(c[3], out_ch, kernel_size=3, padding=1))
        self.encode_modules = nn.ModuleList(encode_list)

        decode_list = []
        for c in cfg["decode"]:
            # c: [height, in_ch, mid_ch, out_ch, RSU4F, side]
            assert len(c) == 6
            decode_list.append(RSU(*c[:4]) if c[4] is False else RSU4F(*c[1:4]))

            if c[5] is True:
                side_list.append(nn.Conv2d(c[3], out_ch, kernel_size=3, padding=1))
        self.decode_modules = nn.ModuleList(decode_list)
        self.side_modules = nn.ModuleList(side_list)
        self.out_conv = nn.Conv2d(self.encode_num * out_ch, out_ch, kernel_size=1)

    def encoder1(self, x: torch.Tensor):
        _, _, h, w = x.shape
        m = self.encode_modules[0]
        x = m(x)
        return x

    def decoder1(self, en1: torch.Tensor):
        m = self.decode_modules[0]
        de1 = m(en1)
        return de1

    def side(self, de1: torch.Tensor):
        _, _, h, w = de1.shape
        m = self.side_modules[0]
        side1 = F.interpolate(m(de1), size=[h, w], mode='bilinear', align_corners=False)
        return side1

    def map(self, img1, img2, se_f):
        """
        Train or Forward for two images
        :param img1: torch.Tensor
        :param img2: torch.Tensor
        :return: output, torch.Tensor
        """
        # Feature extraction c1
        # Decision path
        # Boundary guided filter
        output_origin = torch.sigmoid(1000 * se_f)
        output_blur = self.gaussian(output_origin)
        zeros = torch.zeros_like(output_blur)
        ones = torch.ones_like(output_blur)
        half = ones / 2
        mask_1 = torch.where(output_blur > 0.8, ones, zeros)
        mask_2 = torch.where(output_blur < 0.1, ones, zeros)
        mask_3 = mask_1 * output_blur + mask_2 * (1 - output_blur)
        boundary_map = 1 - torch.abs(2 * (output_blur * mask_3 + (1 - mask_3) * half) - 1)
        temp_fused = img1 * output_origin + (1 - output_origin) * img2
        output_gf = self.guided_filter(temp_fused, output_origin)
        output_bgf = output_gf * boundary_map + output_origin * (1 - boundary_map)
        return output_origin, output_bgf

def u2net_full(out_ch: int = 1):
    cfg = {
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "encode": [[7, 2, 32, 2, False, True]],  # En1
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "decode": [[7, 64, 16, 64, False, False]]  # De1
    }

    return U2Net(cfg, out_ch)


def u2net_lite(out_ch: int = 1):
    cfg = {
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "encode": [[7, 3, 16, 64, False, False],  # En1
                   [6, 64, 16, 64, False, False],  # En2
                   [5, 64, 16, 64, False, False],  # En3
                   [4, 64, 16, 64, False, False],  # En4
                   [4, 64, 16, 64, True, False],  # En5
                   [4, 64, 16, 64, True, True]],  # En6
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "decode": [[4, 128, 16, 64, True, True],  # De5
                   [4, 128, 16, 64, False, True],  # De4
                   [5, 128, 16, 64, False, True],  # De3
                   [6, 128, 16, 64, False, True],  # De2
                   [7, 128, 16, 64, False, True]]  # De1
    }

    return U2Net(cfg, out_ch)


def convert_onnx(m, save_path):
    m.eval()
    x = torch.rand(1, 2, 288, 288, requires_grad=True)

    # export the model
    torch.onnx.export(m,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      save_path,  # where to save the model (can be a file or file-like object)
                      export_params=True,
                      opset_version=11)


if __name__ == '__main__':
    # n_m = RSU(height=7, in_ch=3, mid_ch=12, out_ch=3)
    # convert_onnx(n_m, "RSU7.onnx")
    #
    # n_m = RSU4F(in_ch=3, mid_ch=12, out_ch=3)
    # convert_onnx(n_m, "RSU4F.onnx")

    u2net = u2net_full()
    convert_onnx(u2net, "u2net_full.onnx")

