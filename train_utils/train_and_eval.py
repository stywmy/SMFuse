# #---------------------------------训练结构图--------------------------------------
# import math
# import torch
# from torch import nn
# import numpy as np
# from torch.nn import functional as F
# import train_utils.distributed_utils as utils
# from train_utils.utils import PerceptualLoss
# import imageio
# import pytorch_msssim
#
#
# def criterion(output1, image_jin):
#     # ssim_loss = pytorch_msssim.msssim
#     # ssim_loss_temp = ssim_loss(output1, image_jin, normalize=True)
#     # loss2 = 1 - ssim_loss_temp
#
#     loss2 = F.mse_loss(output1, image_jin)
#     total_loss = loss2
#     return total_loss
#
#
# def evaluate(model, data_loader, device):
#     model.eval()
#     mae_metric = utils.MeanAbsoluteError()
#     f1_metric = utils.F1Score()
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     header = 'Test:'
#     with torch.no_grad():
#         for image_jin, image_yuan, target in metric_logger.log_every(data_loader, 100, header):
#             image_jin, image_yuan, target = image_jin.to(device), image_yuan.to(device), target.to(device)
#             # img = torch.cat([image_jin, image_yuan], 1)
#             # output = model(img)
#             # output = output[0]
#             weight_zeros = torch.zeros(image_jin.shape).to(device)
#             weight_ones = torch.ones(image_jin.shape).to(device)
#             weight_zeros = weight_zeros + 0.1
#             en1_1 = model.encoder1(image_jin)
#             en1_2 = model.encoder1(image_yuan)
#             en1 = torch.where(en1_1 > en1_2, weight_ones, weight_zeros).to(device)
#             en2_1 = model.encoder2(en1_1)
#             en2_2 = model.encoder2(en1_2)
#             weight_zeros = torch.zeros(en2_1.shape).to(device)
#             weight_ones = torch.ones(en2_1.shape).to(device)
#             weight_zeros = weight_zeros + 0.1
#             en2 = torch.where(en2_1 > en2_2, weight_ones, weight_zeros).to(device)
#             en3_1 = model.encoder3(en2_1)
#             en3_2 = model.encoder3(en2_2)
#             weight_zeros = torch.zeros(en3_1.shape).to(device)
#             weight_ones = torch.ones(en3_1.shape).to(device)
#             weight_zeros = weight_zeros + 0.1
#             en3 = torch.where(en3_1 > en3_2, weight_ones, weight_zeros).to(device)
#             de3 = en3
#             de2 = model.decoder2(de3, en2)
#             de1 = model.decoder1(de2, en1)
#             output = model.side(de1, de2, de3)
#             output = output[0]
#
#             mae_metric.update(output, target)
#             f1_metric.update(output, target)
#
#         mae_metric.gather_from_all_processes()
#         f1_metric.reduce_from_all_processes()
#
#     return mae_metric, f1_metric
#
#
# def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler, print_freq=10, scaler=None):
#     model.train()
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
#     header = 'Epoch: [{}]'.format(epoch)
#
#     for image_jin, image_yuan, target in metric_logger.log_every(data_loader, print_freq, header):
#         image_jin, image_yuan, target = image_jin.to(device), image_yuan.to(device), target.to(device)
#         with torch.cuda.amp.autocast(enabled=scaler is not None):
#             en1 = model.encoder1(image_jin)
#             en1 = model.side(en1)
#             output1 = image_jin - en1
#             loss = criterion(output1, image_yuan)
#
#         optimizer.zero_grad()
#         if scaler is not None:
#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
#         else:
#             loss.backward()
#             optimizer.step()
#
#         lr_scheduler.step()
#
#         lr = optimizer.param_groups[0]["lr"]
#         metric_logger.update(loss=loss.item(), lr=lr)
#
#     return metric_logger.meters["loss"].global_avg, lr
#
#
# def create_lr_scheduler(optimizer,
#                         num_step: int,
#                         epochs: int,
#                         warmup=True,
#                         warmup_epochs=1,
#                         warmup_factor=1e-3,
#                         end_factor=1e-6):
#     assert num_step > 0 and epochs > 0
#     if warmup is False:
#         warmup_epochs = 0
#
#     def f(x):
#         """
#         根据step数返回一个学习率倍率因子，
#         注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
#         """
#         if warmup is True and x <= (warmup_epochs * num_step):
#             alpha = float(x) / (warmup_epochs * num_step)
#             # warmup过程中lr倍率因子从warmup_factor -> 1
#             return warmup_factor * (1 - alpha) + alpha
#         else:
#             current_step = (x - warmup_epochs * num_step)
#             cosine_steps = (epochs - warmup_epochs) * num_step
#             # warmup后lr倍率因子从1 -> end_factor
#             return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor
#
#     return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
#
#
# def get_params_groups(model: torch.nn.Module, weight_decay: float = 1e-4):
#     params_group = [{"params": [], "weight_decay": 0.},  # no decay
#                     {"params": [], "weight_decay": weight_decay}]  # with decay
#
#     for name, param in model.named_parameters():
#         if not param.requires_grad:
#             continue  # frozen weights
#
#         if len(param.shape) == 1 or name.endswith(".bias"):
#             # bn:(weight,bias)  conv2d:(bias)  linear:(bias)
#             params_group[0]["params"].append(param)  # no decay
#         else:
#             params_group[1]["params"].append(param)  # with decay
#
#     return params_group

# #---------------------------------训练决策图--------------------------------------
import torch
from torch.nn import functional as F
import train_utils.distributed_utils as utils

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def criterion(inputs, target):
    loss = F.mse_loss(inputs, target)
    return loss

def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler, print_freq=10, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for image_jin, image_yuan, target in metric_logger.log_every(data_loader, print_freq, header):
        image_jin, image_yuan, target = image_jin.to(device), image_yuan.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):

            img = torch.cat([image_jin, image_yuan], 1)
            en1 = model.encoder1(img)
            en1 = model.side(en1)
            loss = criterion(en1, target)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)

    return metric_logger.meters["loss"].global_avg, lr


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            #return warmup_factor * (1 - alpha) + alpha
            return 1
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            # warmup后lr倍率因子从1 -> end_factor
            #return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor
            return 1

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def get_params_groups(model: torch.nn.Module, weight_decay: float = 1e-4):
    params_group = [{"params": [], "weight_decay": 0.},  # no decay
                    {"params": [], "weight_decay": weight_decay}]  # with decay

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if len(param.shape) == 1 or name.endswith(".bias"):
            # bn:(weight,bias)  conv2d:(bias)  linear:(bias)
            params_group[0]["params"].append(param)  # no decay
        else:
            params_group[1]["params"].append(param)  # with decay

    return params_group


