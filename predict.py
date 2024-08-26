# #-------------------------------测试阶段，输出结构图--------------------------------
# import os
# import time
#
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import torch
# from torchvision import models
# from torchvision.transforms import transforms
# import imageio
#
# from src import u2net_full
#
# def main():
#     #threshold = 0.7
#     for j in range(1, 61, 2):
#         print(j)
#         s1 = './MFI_WHU_all/' + str(j) + '.jpg'
#         s2 = './MFI_WHU_all/' + str(j + 1) + '.jpg'
#         s5 = './MFI_WHU_res/' + str(j) + '.jpg'
#         s6 = './MFI_WHU_res/' + str(j+1) + '.jpg'
#         weights_path = "./model_30_structure.pth"
#
#         assert os.path.exists(s1), f"image file {s1} dose not exists."
#         assert os.path.exists(s2), f"image file {s2} dose not exists."
#
#         device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#
#         data_transform = transforms.Compose([
#             transforms.ToTensor(),
#             #transforms.Resize(520),
#             transforms.Normalize(mean=(0.485,),
#                                  std=(0.229,))
#         ])
#         image_jin = cv2.imread(s1, flags=cv2.IMREAD_GRAYSCALE)
#         image_yuan = cv2.imread(s2, flags=cv2.IMREAD_GRAYSCALE)
#         b_image_jin = image_jin
#         b_image_yuan = image_yuan
#         # image_jin = cv2.imread(jin_path, flags=cv2.IMREAD_GRAYSCALE)
#         # image_yuan = cv2.imread(yuan_path, flags=cv2.IMREAD_GRAYSCALE)
#
#         h, w = image_jin.shape[:2]
#
#         b_image_jin = data_transform(b_image_jin)
#         b_image_jin = torch.unsqueeze(b_image_jin, 0).to(device)  # [C, H, W] -> [1, C, H, W]
#
#         b_image_yuan = data_transform(b_image_yuan)
#         b_image_yuan = torch.unsqueeze(b_image_yuan, 0).to(device)  # [C, H, W] -> [1, C, H, W]
#
#         model = u2net_full()
#         weights = torch.load(weights_path, map_location='cpu')
#         if "model" in weights:
#             model.load_state_dict(weights["model"])
#         else:
#             model.load_state_dict(weights)
#         model.to(device)
#         model.eval()
#
#         with torch.no_grad():
#             en1 = model.encoder1(b_image_jin)
#             en1 = model.side(en1)
#
#             en2 = model.encoder1(b_image_yuan)
#             en2 = model.side(en2)
#
#             en1 = torch.squeeze(en1).to("cpu").numpy()  # [1, 1, H, W] -> [H, W]
#             en2 = torch.squeeze(en2).to("cpu").numpy()  # [1, 1, H, W] -> [H, W]
#             imageio.imsave(s5, en1)
#             imageio.imsave(s6, en2)
#
# if __name__ == '__main__':
#     main()

#------------------------输出结构图构建第二阶段训练集--------------------------------
# import os
# import time
# from torchvision import models
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import torch
# from torchvision.transforms import transforms
# import imageio
#
# from src import u2net_full
#
# def main():
#     #threshold = 0.7
#     s1 = "/data/Disk_A/tianyu/u2net/DATA-TR/DATA-TR-jin/"
#     s2 = "/data/Disk_A/tianyu/u2net/DATA-TR/DATA-TR-yuan/"
#     sm = "/data/Disk_A/tianyu/u2net/DATA-TR/DATA-TR-mask/"
#     names = os.listdir(s1)
#     count = 1
#     for name in names:
#         file_path1 = s1 + name
#         file_path2 = s2 + name
#         file_path3 = sm + name
#         # s5 = "/data/Disk_A/tianyu/u2net/DATA/DATA-jin-res/"
#         # s6 = "/data/Disk_A/tianyu/u2net/DATA/DATA-yuan-res/"
#         # smm = "/data/Disk_A/tianyu/u2net/DATA/DATA-mask-res/"
#         s5 = "/data/Disk_A/tianyu/u2net/DATA-TR/DATA-jin-test-res-szw/"
#         s6 = "/data/Disk_A/tianyu/u2net/DATA-TR/DATA-yuan-test-res-szw/"
#         smm = "/data/Disk_A/tianyu/u2net/DATA-TR/DATA-mask-test-szw/"
#         # s11 = "/data/Disk_A/tianyu/u2net/DATA-TR/DATA-jin-test-res-pipei/"
#         # s22 = "/data/Disk_A/tianyu/u2net/DATA-TR/DATA-yuan-test-res-pipei/"
#         to_save_path1 = s5 + str(count) + '.jpg'
#         to_save_path2 = s6 + str(count) + '.jpg'
#         to_save_path3 = smm + str(count) + '.jpg'
#         # to_save_path4 = s11 + str(count) + '.jpg'
#         # to_save_path5 = s22 + str(count) + '.jpg'
#         weights_path = "./model_30_structure.pth"
#         # jin_path = "./dog.jpg"
#
#         assert os.path.exists(s1), f"image file {s1} dose not exists."
#         assert os.path.exists(s2), f"image file {s2} dose not exists."
#
#         device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#
#         data_transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Resize(520),
#             transforms.Normalize(mean=(0.485,),
#                                  std=(0.229,))
#         ])
#         image_jin = cv2.imread(file_path1, flags=cv2.IMREAD_GRAYSCALE)
#         image_yuan = cv2.imread(file_path2, flags=cv2.IMREAD_GRAYSCALE)
#         image_mask = cv2.imread(file_path3, flags=cv2.IMREAD_GRAYSCALE)
#         b_image_jin = image_jin
#         b_image_yuan = image_yuan
#         # image_jin = cv2.imread(jin_path, flags=cv2.IMREAD_GRAYSCALE)
#         # image_yuan = cv2.imread(yuan_path, flags=cv2.IMREAD_GRAYSCALE)
#
#         h, w = image_jin.shape[:2]
#
#         b_image_jin = data_transform(b_image_jin)
#         b_image_jin = torch.unsqueeze(b_image_jin, 0).to(device)  # [C, H, W] -> [1, C, H, W]
#         # b_image_jinn = torch.squeeze(b_image_jin).to("cpu").numpy()
#         # imageio.imsave(to_save_path4, b_image_jinn)
#
#         b_image_yuan = data_transform(b_image_yuan)
#         b_image_yuan = torch.unsqueeze(b_image_yuan, 0).to(device)  # [C, H, W] -> [1, C, H, W]
#         # b_image_yuann = torch.squeeze(b_image_yuan).to("cpu").numpy()
#         # imageio.imsave(to_save_path5, b_image_yuann)
#
#         image_mask = data_transform(image_mask)
#         image_mask = torch.unsqueeze(image_mask, 0).to(device)  # [C, H, W] -> [1, C, H, W]
#         image_mask = torch.squeeze(image_mask).to("cpu").numpy()
#         imageio.imsave(to_save_path3, image_mask)
#
#         model = u2net_full()
#         weights = torch.load(weights_path, map_location='cpu')
#         if "model" in weights:
#             model.load_state_dict(weights["model"])
#         else:
#             model.load_state_dict(weights)
#         model.to(device)
#         model.eval()
#
#         with torch.no_grad():
#             en1 = model.encoder1(b_image_jin)
#             en1 = model.side(en1)
#             #en1 = en1 + b_image_jin
#             en2 = model.encoder1(b_image_yuan)
#             en2 = model.side(en2)
#             #en2 = en2 + b_image_yuan
#
#             # vgg_model = models.vgg19(pretrained=True)
#             # vgg_model = vgg_model.cuda(device)
#             # vggFeatures = []
#             # vggFeatures.append(vgg_model.features[:4])  # 64
#             # vggFeatures.append(vgg_model.features[:9])  # 32
#             # vggFeatures.append(vgg_model.features[:18])  # 16
#             # vggFeatures.append(vgg_model.features[:27])  # 8
#             # vggFeatures.append(vgg_model.features[:36])  # 4
#             # for i in range(0, 5):
#             #     for parm in vggFeatures[i].parameters():
#             #         parm.requires_grad = False;
#
#             # img_irdup = torch.cat([en1, en1, en1], 1);
#             # img_vidup = torch.cat([en2, en2, en2], 1);
#             #
#             # g_ir = vggFeatures[0](img_irdup);
#             # g_vi = vggFeatures[0](img_vidup);
#             # en1 = torch.squeeze(g_ir, 0)
#             # en1 = torch.sum(en1, 0)
#             en1 = torch.squeeze(en1, 0)
#             en1 = torch.squeeze(en1, 0)
#             en2 = torch.squeeze(en2, 0)
#             en2 = torch.squeeze(en2, 0)
#             en1 = en1.to('cpu').detach().numpy()
#             # en2 = torch.squeeze(g_vi, 0)
#             # en2 = torch.sum(en2, 0)
#             en2 = en2.to('cpu').detach().numpy()
#
#             #en1 = torch.squeeze(en1).to("cpu").numpy()  # [1, 1, H, W] -> [H, W]
#             #en2 = torch.squeeze(en2).to("cpu").numpy()  # [1, 1, H, W] -> [H, W]
#             #pred2 = np.where(pred2 > threshold, 1, 0)
#             #imageio.imsave(s3, pred1)
#             imageio.imsave(to_save_path1, en1)
#             imageio.imsave(to_save_path2, en2)
#             print("------conut：", count)
#             count += 1
#             # pred = torch.squeeze(pred).to("cpu").numpy()  # [1, 1, H, W] -> [H, W]
#             #sty = 1
#
#
# if __name__ == '__main__':
#     main()

#---------------------------------------输出二分类的结果----------------------------------------
import os
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import transforms
import imageio

from src import u2net_full

def main():
    threshold = 0.5
    for j in range(1, 61, 2):
        print(j)
        s1 = './MFFW2-res/' + str(j) + '.jpg'
        s2 = './MFFW2-res/' + str(j + 1) + '.jpg'
        s4 = './MFFW2-output-no/' + str(int((j + 1) / 2)) + '.jpg' #保存网络直接输出的决策图
        s5 = './MFFW2-output/' + str(int((j + 1) / 2)) + '.jpg'  #保存初步的的决策图
        weights_path = "./model_30.pth"

        assert os.path.exists(s1), f"image file {s1} dose not exists."
        assert os.path.exists(s2), f"image file {s2} dose not exists."

        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485,),
                                 std=(0.229,))
        ])
        image_jin = cv2.imread(s1, flags=cv2.IMREAD_GRAYSCALE)
        image_yuan = cv2.imread(s2, flags=cv2.IMREAD_GRAYSCALE)
        b_image_jin = image_jin
        b_image_yuan = image_yuan

        h, w = image_jin.shape[:2]

        b_image_jin = data_transform(b_image_jin)
        b_image_jin = torch.unsqueeze(b_image_jin, 0).to(device)  # [C, H, W] -> [1, C, H, W]

        b_image_yuan = data_transform(b_image_yuan)
        b_image_yuan = torch.unsqueeze(b_image_yuan, 0).to(device)  # [C, H, W] -> [1, C, H, W]

        model = u2net_full()
        weights = torch.load(weights_path, map_location='cpu')
        if "model" in weights:
            model.load_state_dict(weights["model"])
        else:
            model.load_state_dict(weights)
        model.to(device)
        model.eval()

        with torch.no_grad():

            start_time = time.time()
            img = torch.cat([b_image_jin, b_image_yuan], 1)
            en1 = model.encoder1(img)
            en1 = model.side(en1)
            pred = torch.squeeze(en1).to("cpu").numpy()  # [1, 1, H, W] -> [H, W]
            imageio.imsave(s4, pred)
            pred = np.where(pred > threshold, 1, 0)

            end_time = time.time()  # 记录结束时间
            run_time = end_time - start_time  # 计算运行时间（单位为秒）
            print("程序运行时间为：", run_time)

            imageio.imsave(s5, pred)

if __name__ == '__main__':
    main()