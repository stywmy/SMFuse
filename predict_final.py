# #------------------------------------得到最终融合结果---------------------------------------
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import skimage
from skimage import morphology

mat = np.array(
    [[65.481, 128.553, 24.966],
     [-37.797, -74.203, 112.0],
     [112.0, -93.786, -18.214]])
mat_inv = np.linalg.inv(mat)
offset = np.array([16, 128, 128])

def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def ycbcr2rgb(ycbcr_img):
    rgb_img = np.zeros(ycbcr_img.shape, dtype=np.uint8)
    for x in range(ycbcr_img.shape[0]):
        for y in range(ycbcr_img.shape[1]):
            [r, g, b] = ycbcr_img[x, y, :]
            rgb_img[x, y, :] = np.maximum(0, np.minimum(255,
                                                        np.round(np.dot(mat_inv, ycbcr_img[x, y, :] - offset) * 255.0)))
    return rgb_img
def rgb2ycbcr(rgb_img):
    ycbcr_img = np.zeros(rgb_img.shape, dtype=np.uint8)
    for x in range(rgb_img.shape[0]):
        for y in range(rgb_img.shape[1]):
            ycbcr_img[x, y, :] = np.round(np.dot(mat, rgb_img[x, y, :] * 1.0 / 255) + offset)
    return ycbcr_img

def fusion_channel_sf(f1):
    c, h, w = f1.shape
    dm_tensor = f1.cpu()
    dm = dm_tensor.squeeze().cpu().detach().numpy().astype(np.int32)
    # se = skimage.morphology.disk(3)  # 'disk' kernel with ks size for structural element
    se = skimage.morphology.disk(1)  # 'disk' kernel with ks size for structural element
    dm = skimage.morphology.binary_opening(dm, se)#开运算,先腐蚀再膨胀，可以消除小物体或小斑块
    #dm = skimage.morphology.opening(dm, se)  # 开运算,先腐蚀再膨胀，可以消除小物体或小斑块
    dm = morphology.remove_small_holes(dm == 0, 0.002 * h * w)
    #dm = morphology.remove_small_holes(dm == 0, 0.00001 * h * w)
    dm = np.where(dm, 0, 1)
    dm = skimage.morphology.binary_closing(dm, se)#闭运算,先膨胀再腐蚀，可用来填充孔洞
    #dm = skimage.morphology.closing(dm, se)  # 闭运算,先膨胀再腐蚀，可用来填充孔洞
    dm = morphology.remove_small_holes(dm == 1, 0.002 * h * w)
    #dm = morphology.remove_small_holes(dm == 1, 0.00001 * h * w)
    dm = np.where(dm, 1, 0)
    dm = torch.Tensor(dm)
    dm_np2 = dm.squeeze().cpu().detach().numpy().astype(np.float64)
    return dm_np2

def box_filter(imgSrc, r):
    """
    Definition imDst(x, y)=sum(sum(imSrc(x-r:x+r,y-r:y+r)));
    :param imgSrc: np.array, image
    :param r: int, radius
    :return: imDst: np.array. result of calculation
    """
    if imgSrc.ndim == 2:
        h, w = imgSrc.shape[:2]
        imDst = np.zeros(imgSrc.shape[:2])

        # cumulative sum over h axis
        imCum = np.cumsum(imgSrc, axis=0)
        # difference over h axis
        imDst[0: r + 1] = imCum[r: 2 * r + 1]
        imDst[r + 1: h - r] = imCum[2 * r + 1: h] - imCum[0: h - 2 * r - 1]
        imDst[h - r: h, :] = np.tile(imCum[h - 1, :], [r, 1]) - imCum[h - 2 * r - 1: h - r - 1, :]

        # cumulative sum over w axis
        imCum = np.cumsum(imDst, axis=1)

        # difference over w axis
        imDst[:, 0: r + 1] = imCum[:, r: 2 * r + 1]
        imDst[:, r + 1: w - r] = imCum[:, 2 * r + 1: w] - imCum[:, 0: w - 2 * r - 1]
        imDst[:, w - r: w] = np.tile(np.expand_dims(imCum[:, w - 1], axis=1), [1, r]) - \
                             imCum[:, w - 2 * r - 1: w - r - 1]
    else:
        h, w = imgSrc.shape[:2]
        imDst = np.zeros(imgSrc.shape)

        # cumulative sum over h axis
        imCum = np.cumsum(imgSrc, axis=0)
        # difference over h axis
        imDst[0: r + 1] = imCum[r: 2 * r + 1]
        imDst[r + 1: h - r, :] = imCum[2 * r + 1: h, :] - imCum[0: h - 2 * r - 1, :]
        imDst[h - r: h, :] = np.tile(imCum[h - 1, :], [r, 1, 1]) - imCum[h - 2 * r - 1: h - r - 1, :]

        # cumulative sum over w axis
        imCum = np.cumsum(imDst, axis=1)

        # difference over w axis
        imDst[:, 0: r + 1] = imCum[:, r: 2 * r + 1]
        imDst[:, r + 1: w - r] = imCum[:, 2 * r + 1: w] - imCum[:, 0: w - 2 * r - 1]
        imDst[:, w - r: w] = np.tile(np.expand_dims(imCum[:, w - 1], axis=1), [1, r, 1]) - \
                             imCum[:, w - 2 * r - 1: w - r - 1]
        return imDst

def guided_filter(I, p, r, eps=0.1):
    """
    Guided Filter
    :param I: np.array, guided image
    :param p: np.array, input image
    :param r: int, radius
    :param eps: float
    :return: np.array, filter result
    """
    h, w = I.shape[:2]
    if I.ndim == 2:
        N = box_filter(np.ones((h, w)), r)
    else:
        N = box_filter(np.ones((h, w, 1)), r)
    mean_I = box_filter(I, r) / N
    mean_p = box_filter(p, r) / N
    mean_Ip = box_filter(I * p, r) / N
    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = box_filter(I * I, r) / N
    var_I = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + eps)

    if I.ndim == 2:
        b = mean_p - a * mean_I
        mean_a = box_filter(a, r) / N
        mean_b = box_filter(b, r) / N
        q = mean_a * I + mean_b
    else:
        b = mean_p - np.expand_dims(np.sum((a * mean_I), 2), 2)
        mean_a = box_filter(a, r) / N
        mean_b = box_filter(b, r) / N
        q = np.expand_dims(np.sum(mean_a * I, 2), 2) + mean_b
        return q

for j in range(1, 61, 2):
    print(j)
    s1 = './MFFW/' + str(j) + '.jpg'
    s2 = './MFFW/' + str(j + 1) + '.jpg'
    s3 = './MFFW_results/' + str(int((j+1)/2)) + '.jpg' #保存最终的融合图像下·
    # s6 = './MFFW_results_test/' + str(int((j + 1) / 2)) + '.jpg'
    s4 = './MFFW_map/' + str(int((j+1)/2)) + '.jpg' #保存最终细化后的决策图
    s5 = './MFFW2-output/' + str(int((j+1)/2)) + '.jpg'
    im_b_y_1 = plt.imread(s1)
    im_b_y_2 = plt.imread(s2)
    im_b_y_5 = plt.imread(s5)
    inp = im_b_y_1
    img1 = im_b_y_1
    img2 = im_b_y_2

    if inp.ndim == 3:
        a_5 = cv2.imread(s5, flags=cv2.IMREAD_GRAYSCALE)
        im_b_y_5 = a_5
        im_b_y_5 = im_b_y_5.astype(float)
        im_b_y_5 = im_b_y_5 / 1.
        # result = np.zeros([im_b_y.shape[0],im_b_y.shape[1],im_b_y.shape[2]],dtype=np.float32)
        for i in range(0, 1):
            # print(i)
            im_input_1 = im_b_y_5
            im_input_1 = Variable(torch.from_numpy(im_input_1).float()).view(1, -1, im_input_1.shape[0],
                                                                             im_input_1.shape[1])
            im_input_1 = im_input_1.cuda()
            HR_1 = im_input_1
            HR_1 = torch.sum(HR_1, dim=0)
            HR_1 = HR_1.cpu()
            dm2 = fusion_channel_sf(HR_1)  # dm2是一个二值图
            # dm2 = dm2.squeeze().cpu().detach().numpy().astype(np.float64)
            # import imageio
            # imageio.imsave("dm2.png", dm2)
            m = dm2.shape[0]  # m是行数
            n = dm2.shape[1]  # n是列数
            a = dm2
            b = np.ones((m, n))
            for chang in range(0, m):
                for kuan in range(0, n):
                    if (chang + 2) > m - 1 or (kuan + 2) > n - 1 or (chang - 2) < 0 or (kuan - 2) < 0:
                        b[chang, kuan] = 0
                    else:
                        sum = a[chang - 1, kuan - 1] + a[chang - 1, kuan] + a[chang - 1, kuan + 1] + \
                              a[chang, kuan - 1] + a[chang, kuan] + a[chang, kuan + 1] + \
                              a[chang + 1, kuan - 1] + a[chang + 1, kuan] + a[chang + 1, kuan + 1] + \
                              a[chang - 2, kuan - 2] + a[chang - 2, kuan - 1] + a[chang - 2, kuan] + \
                              a[chang - 2, kuan + 1] + a[chang - 2, kuan + 2] + a[chang - 1, kuan - 2] + \
                              a[chang - 1, kuan + 2] + a[chang, kuan - 2] + a[chang, kuan + 2] + \
                              a[chang + 1, kuan - 2] + a[chang + 1, kuan + 2] + a[chang + 2, kuan - 2] + \
                              a[chang + 2, kuan - 1] + a[chang + 2, kuan] + a[chang + 2, kuan + 1] + \
                              a[chang + 2, kuan + 2]
                        if sum == 0 or sum == 25:
                            b[chang, kuan] = 0

            if inp.ndim == 3:
                # dm = np.expand_dims(dm, axis=2)
                dm2 = np.expand_dims(dm2, axis=2)  # 二维变三维
                b = np.expand_dims(b, axis=2)
            # sty用来测试
            # look_b = np.clip(b, 0, 255).astype(np.uint8)
            # plt.imsave(s5, look_b)
            # import imageio
            # imageio.imsave("b_Pretrain.png", b)
            temp_fused = img1 * dm2 + img2 * (1 - dm2)
            temp_fused = rgb2gray(temp_fused)
            temp_fused = np.expand_dims(temp_fused, axis=2)
            dm2_filter = guided_filter(temp_fused, dm2, r=2, eps=0.1)

            dm_final = np.where(b == 1, dm2_filter, dm2)
            fused = img1 * 1.0 * dm_final + img2 * 1.0 * (1 - dm_final)
            fused1 = fused * 1.0 * (1 - b)
            fused1 = fused1 + 255 * b
            fused1 = np.clip(fused1, 0, 255).astype(np.uint8)
            result = fused1
            #plt.imsave(s6, result)

            fused = np.clip(fused, 0, 255).astype(np.uint8)
            result = fused
            plt.imsave(s3, result)
            dm_write = (dm_final * 255).astype(np.uint8)
            decisionmap = np.repeat(dm_write, 3, axis=2)
            #imageio.imsave(s4, decisionmap)
            plt.imsave(s4, decisionmap)
           #plt.imsave(s3, result)