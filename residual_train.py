import argparse, os
import torch
from torch.autograd import Variable
import numpy as np
from mynet import MyNet
from torch.utils.data import  DataLoader
from data_loader.train_data import LoadData
from tqdm import tqdm
from torch import optim
import torch.nn.functional as F
import pytorch_msssim
#from torchstat import stat
import torch
import gc
import scipy.io as scio
# torch.cuda.set_device(0)
# gc.collect()
# torch.cuda.empty_cache()

# import sys
# import importlib
# importlib.reload(sys)
# sys.setdefaultencoding('utf-8')
parser = argparse.ArgumentParser(description="PyTorch VDSR Eval")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
# parser.add_argument("--model", default="model/model_epoch_50.pth", type=str, help="model path")
parser.add_argument('--dataset_path', metavar='DIR', default='data_coco',
                        help='path to dataset (default: imagenet)')
parser.add_argument('--save_path', default='pretrained_313')  # 模型存储路径
parser.add_argument('--start_epoch', default=1, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', default=101, type=int, metavar='N',
                        help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=16, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
parser.add_argument("--dataset", default="Myset", type=str, help="dataset name, Default: Set5")
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
parser.add_argument("--pretrained", default="./model_epoch_26.pth", type=str,
                    help="path to pretrained model (default: none)")

opt = parser.parse_args()
cuda = opt.cuda

train_dataset = LoadData(opt.dataset_path)
train_loader = DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True,
        pin_memory=True)

offset = np.array([16, 128, 128])

if cuda:
    print("=> use gpu id: '{}'".format(opt.gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    if not torch.cuda.is_available():
        raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

# model = torch.load(opt.model, map_location=lambda storage, loc: storage)["model"]

model = MyNet()
model.cuda()
Loss_all = []
Loss_all2 = []

optimizer = optim.SGD(model.parameters(), lr = opt.lr, momentum=0.9, weight_decay=1e-4)
for epoch in range(opt.start_epoch, opt.epochs):
    # if epoch < opt.epochs // 2:
    #     lr = opt.lr
    # else:
    #     lr = opt.lr * (opt.epochs - epoch) / (opt.epochs - opt.epochs // 2)

    if epoch < 6:
        lr = opt.lr
    if 6 <= epoch < 30:
        lr = 0.01
    if 30 <= epoch < 70:
        lr = 0.005
    if 70 <= epoch < 90:
        lr = 0.001
    if epoch >= 90:
        lr = 0.0005

    # 修改学习率
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    model.train()#训练
    ssim_loss = pytorch_msssim.msssim
    #stat(model, (3, 224, 224))
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    train_tqdm = tqdm(train_loader, total=len(train_loader))
    ssim_loss_value = 0.
    loss_total = 0.
    loss_total2 = 0.
    for compare_Xi, compare_Yi in train_tqdm:
        # Yi_image = Yi_image.detach().cpu().numpy()
        optimizer.zero_grad()
        model = model.cuda()
        compare_Xi = compare_Xi.cuda()
        compare_Yi = compare_Yi.cuda()
        # if cuda:
        #     model = model.cuda()
        #     compare_Xi = compare_Xi.cuda()
        #     compare_Yi = compare_Yi.cuda()
        # else:
        #     model = model.cpu()
        compare_Xi = compare_Xi.float()
        compare_Yi = compare_Yi.float()
        f_Xi = model.forward(compare_Xi, 'train')
        loss = F.mse_loss(f_Xi, compare_Yi)
        ssim_loss_temp = ssim_loss(compare_Yi, f_Xi, normalize=True)
        if loss > 1:
            length = len(str(int(loss)))
            quan = pow(10,length)
            ssim_loss_value = quan*(1 - ssim_loss_temp)
        else:
            ssim_loss_value = 1 - ssim_loss_temp
        loss_total = loss_total + loss.item() + ssim_loss_value.item()
        if (epoch > 1):
            loss_total2 = loss_total2 + loss.item() + ssim_loss_value.item()
        loss_sum = loss + ssim_loss_value
        train_tqdm.set_postfix(epoch=epoch, loss_total=loss_total)
        loss_sum.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

    Loss_all.append(loss_total)
    loss_data_all = np.array(Loss_all)
    data_mat = 'data_mat.mat'
    scio.savemat(data_mat, {'Loss': loss_data_all})
    # showLossChart(mat文件的路径,输出曲线的位置);
    showLossChart(data_mat, 'loss.png')

    if (epoch > 1):
        Loss_all2.append(loss_total2)
        loss_data_all2 = np.array(Loss_all2)
        data_mat2 = 'data_mat2.mat'
        scio.savemat(data_mat2, {'Loss': loss_data_all2})
        # showLossChart(mat文件的路径,输出曲线的位置);
        showLossChart(data_mat2, 'loss2.png')
    torch.save(model.state_dict(), f'{opt.save_path}/model_epoch_{epoch}.pth')
#---------------------------------------------------------------------------------------------
# #删除掉SSIM损失函数
# import argparse, os
# import torch
# from torch.autograd import Variable
# import numpy as np
# from mynet import MyNet
# from torch.utils.data import  DataLoader
# from data_loader.train_data import LoadData
# from tqdm import tqdm
# from torch import optim
# import torch.nn.functional as F
# import pytorch_msssim
# #from torchstat import stat
# import torch
# import gc
# import scipy.io as scio
# from testMat import showLossChart
# torch.cuda.set_device(1)
# gc.collect()
# torch.cuda.empty_cache()
#
# # import sys
# # import importlib
# # importlib.reload(sys)
# # sys.setdefaultencoding('utf-8')
# parser = argparse.ArgumentParser(description="PyTorch VDSR Eval")
# parser.add_argument("--cuda", action="store_true", help="use cuda?")
# # parser.add_argument("--model", default="model/model_epoch_50.pth", type=str, help="model path")
# parser.add_argument('--dataset_path', metavar='DIR', default='data_coco',
#                         help='path to dataset (default: imagenet)')
# parser.add_argument('--save_path', default='pretrained_126')  # 模型存储路径
# parser.add_argument('--start_epoch', default=1, type=int, metavar='N',
#                         help='manual epoch number (useful on restarts)')
# parser.add_argument('--epochs', default=101, type=int, metavar='N',
#                         help='number of total epochs to run')
# parser.add_argument('-b', '--batch_size', default=16, type=int,
#                         metavar='N',
#                         help='mini-batch size (default: 256), this is the total '
#                              'batch size of all GPUs on the current node when '
#                              'using Data Parallel or Distributed Data Parallel')
# parser.add_argument("--dataset", default="Myset", type=str, help="dataset name, Default: Set5")
# parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
#                         metavar='LR', help='initial learning rate', dest='lr')
# parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
# parser.add_argument("--pretrained", default="./model_epoch_26.pth", type=str,
#                     help="path to pretrained model (default: none)")
#
# opt = parser.parse_args()
# cuda = opt.cuda
#
# train_dataset = LoadData(opt.dataset_path)
# train_loader = DataLoader(
#         train_dataset, batch_size=opt.batch_size, shuffle=True,
#         pin_memory=True)
#
# offset = np.array([16, 128, 128])
#
# if cuda:
#     print("=> use gpu id: '{}'".format(opt.gpus))
#     os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
#     if not torch.cuda.is_available():
#         raise Exception("No GPU found or Wrong gpu id, please run without --cuda")
#
# # model = torch.load(opt.model, map_location=lambda storage, loc: storage)["model"]
#
# model = MyNet()
# model.cuda()
# Loss_all = []
# Loss_all2 = []
# optimizer = optim.SGD(model.parameters(), lr = opt.lr, momentum=0.9, weight_decay=1e-4)
# for epoch in range(opt.start_epoch, opt.epochs):
#     # if epoch < opt.epochs // 2:
#     #     lr = opt.lr
#     # else:
#     #     lr = opt.lr * (opt.epochs - epoch) / (opt.epochs - opt.epochs // 2)
#     if epoch < 3:
#         lr = opt.lr
#     if 3 <= epoch < 20:
#         lr = 0.01
#     if 20 <= epoch < 60:
#         lr = 0.005
#     if 60 <= epoch < 80:
#         lr = 0.001
#     if epoch >= 80:
#         lr = 0.0005
#
#     # if epoch < 3:
#     #     lr = opt.lr
#     # if 3 <= epoch < 40:
#     #     lr = 0.01
#     # if 40 <= epoch < 80:
#     #     lr = 0.005
#     # if epoch >= 80:
#     #     lr = 0.001
#
#     # 修改学习率
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
#
#     model.train()#训练
#     ssim_loss = pytorch_msssim.msssim
#     #stat(model, (3, 224, 224))
#     total = sum([param.nelement() for param in model.parameters()])
#     print("Number of parameter: %.2fM" % (total / 1e6))
#     train_tqdm = tqdm(train_loader, total=len(train_loader))
#     ssim_loss_value = 0.
#     loss_total = 0.
#     loss_total2 = 0.
#     for compare_Xi, compare_Yi in train_tqdm:
#         # Yi_image = Yi_image.detach().cpu().numpy()
#         optimizer.zero_grad()
#         model = model.cuda()
#         compare_Xi = compare_Xi.cuda()
#         compare_Yi = compare_Yi.cuda()
#         # if cuda:
#         #     model = model.cuda()
#         #     compare_Xi = compare_Xi.cuda()
#         #     compare_Yi = compare_Yi.cuda()
#         # else:
#         #     model = model.cpu()
#         compare_Xi = compare_Xi.float()
#         compare_Yi = compare_Yi.float()
#         f_Xi = model.forward(compare_Xi, 'train')
#         loss = F.mse_loss(f_Xi, compare_Yi)
#         loss_total = loss_total + loss.item()
#         if(epoch>1):
#             loss_total2 = loss_total2 + loss.item()
#         loss_sum = loss
#         train_tqdm.set_postfix(epoch=epoch, loss_total=loss_total)
#         loss_sum.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
#         optimizer.step()
#
#     Loss_all.append(loss_total)
#     loss_data_all = np.array(Loss_all)
#     data_mat = 'data_mat.mat'
#     scio.savemat(data_mat, {'Loss': loss_data_all})
#     # showLossChart(mat文件的路径,输出曲线的位置);
#     showLossChart(data_mat,'loss.png')
#
#     if(epoch>1):
#         Loss_all2.append(loss_total2)
#         loss_data_all2 = np.array(Loss_all2)
#         data_mat2 = 'data_mat2.mat'
#         scio.savemat(data_mat2, {'Loss': loss_data_all2})
#         # showLossChart(mat文件的路径,输出曲线的位置);
#         showLossChart(data_mat2, 'loss2.png')
#     torch.save(model.state_dict(), f'{opt.save_path}/model_epoch_{epoch}.pth')