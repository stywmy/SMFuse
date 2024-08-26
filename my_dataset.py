# #-------------------------------------------只是用于结构图------------------------------------
# import os
#
# import cv2
# import torch.utils.data as data
#
#
# class DUTSDataset(data.Dataset):
#     def __init__(self, root: str, train: bool = True, transforms=None):
#         assert os.path.exists(root), f"path '{root}' does not exist."
#         if train:
#             self.image_jin_root = os.path.join("/data/Disk_A/tianyu/u2net/MSCOCO2014/", "test_clear")
#             self.image_yuan_root = os.path.join("/data/Disk_A/tianyu/u2net/MSCOCO2014/", "test_mohu")
#             self.target_root = os.path.join("/data/Disk_A/tianyu/u2net/MSCOCO2014/", "test_mohu")
#             # self.image_jin_root = os.path.join(root, "DATA-TR", "DATA-TR-jin-sf")
#             # self.image_yuan_root = os.path.join(root, "DATA-TR", "DATA-TR-yuan-sf")
#             # self.target_root = os.path.join(root, "DATA-TR", "DATA-TR-mask")
#             # self.image_jin_root = os.path.join(root, "DATA-TE", "DATA-TE-jin-sf")
#             # self.image_yuan_root = os.path.join(root, "DATA-TE", "DATA-TE-yuan-sf")
#             # self.target_root = os.path.join(root, "DATA-TE", "DATA-TE-mask")
#         else:
#             self.image_jin_root = os.path.join(root, "DUTS-TE", "DUTS-TE-jin")
#             self.image_yuan_root = os.path.join(root, "DUTS-TE", "DUTS-TE-jin")
#             self.target_root = os.path.join(root, "DUTS-TE", "DUTS-TE-jin")
#         assert os.path.exists(self.image_jin_root), f"path '{self.image_jin_root}' does not exist."
#         assert os.path.exists(self.image_yuan_root), f"path '{self.image_yuan_root}' does not exist."
#         assert os.path.exists(self.target_root), f"path '{self.target_root}' does not exist."
#
#         image_jin_names = [p for p in os.listdir(self.image_jin_root) if p.endswith(".jpg")]
#         image_yuan_names = [p for p in os.listdir(self.image_yuan_root) if p.endswith(".jpg")]
#         image_target_names = [p for p in os.listdir(self.target_root) if p.endswith(".jpg")]
#         # image_names = [p for p in os.listdir(self.image_root) if p.endswith(".jpg")]
#         # mask_names = [p for p in os.listdir(self.mask_root) if p.endswith(".png")]
#         assert len(image_jin_names) > 0, f"not find any images in {self.image_jin_root}."
#
#         # check images and mask
#         # re_mask_names = []
#         # for p in image_names:
#         #     mask_name = p.replace(".jpg", ".png")
#         #     assert mask_name in mask_names, f"{p} has no corresponding mask."
#         #     re_mask_names.append(mask_name)
#         # mask_names = re_mask_names
#
#         self.images_jin_path = [os.path.join(self.image_jin_root, n) for n in image_jin_names]
#         self.images_yuan_path = [os.path.join(self.image_yuan_root, n) for n in image_jin_names]
#         self.images_target_path = [os.path.join(self.target_root, n) for n in image_target_names]
#         # self.images_path = [os.path.join(self.image_root, n) for n in image_names]
#         # self.masks_path = [os.path.join(self.mask_root, n) for n in mask_names]
#
#         self.transforms = transforms
#
#     def __getitem__(self, idx):
#         images_jin_path = self.images_jin_path[idx]
#         images_yuan_path = self.images_yuan_path[idx]
#         images_target_path = self.images_target_path[idx]
#
#         images_jin = cv2.imread(images_jin_path, flags=cv2.IMREAD_GRAYSCALE)
#         images_yuan = cv2.imread(images_yuan_path, flags=cv2.IMREAD_GRAYSCALE)
#         target = cv2.imread(images_target_path, flags=cv2.IMREAD_GRAYSCALE)
#         assert images_jin is not None, f"failed to read image: {images_jin_path}"
#         h, _ = images_jin.shape
#         assert target is not None, f"failed to read mask: {images_target_path}"
#
#         if self.transforms is not None:
#             image_jin, image_yuan, target = self.transforms(images_jin, images_yuan, target)
#
#         return image_jin, image_yuan, target
#
#     def __len__(self):
#         return len(self.images_jin_path)
#
#     @staticmethod
#     def collate_fn(batch):
#
#         images_jin, images_yuan, targets = list(zip(*batch))
#         batched_imgs_jin = cat_list(images_jin, fill_value=0)
#         batched_imgs_yuan = cat_list(images_yuan, fill_value=0)
#         batched_targets = cat_list(targets, fill_value=0)
#
#         return batched_imgs_jin, batched_imgs_yuan, batched_targets
#
#
# def cat_list(images, fill_value=0):
#     max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
#     batch_shape = (len(images),) + max_size
#     batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
#     for img, pad_img in zip(images, batched_imgs):
#         pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
#     return batched_imgs
#
#
# if __name__ == '__main__':
#     train_dataset = DUTSDataset("./", train=True)
#     print(len(train_dataset))
#
#     val_dataset = DUTSDataset("./", train=False)
#     print(len(val_dataset))
#
#     i, t = train_dataset[0]

#----------------------------------用于训练决策图----------------------------------------
import os

import cv2
import torch.utils.data as data


class DUTSDataset(data.Dataset):
    def __init__(self, root: str, train: bool = True, transforms=None):
        assert os.path.exists(root), f"path '{root}' does not exist."
        if train:
            self.image_jin_root = os.path.join("/data/Disk_A/tianyu/u2net/DATA-TR/", "DATA-TR-jin-gsres")
            self.image_yuan_root = os.path.join("/data/Disk_A/tianyu/u2net/DATA-TR/", "DATA-TR-yuan-gsres")
            self.target_root = os.path.join("/data/Disk_A/tianyu/u2net/DATA-TR/", "DATA-TR-mask-gsres")
        else:
            self.image_jin_root = os.path.join("/data/Disk_A/tianyu/u2net/DATA-TR/", "DATA-jin-test-res")
            self.image_yuan_root = os.path.join("/data/Disk_A/tianyu/u2net/DATA-TR/", "DATA-yuan-test-res")
            self.target_root = os.path.join("/data/Disk_A/tianyu/u2net/DATA-TR/", "DATA-mask-test-res")
        assert os.path.exists(self.image_jin_root), f"path '{self.image_jin_root}' does not exist."
        assert os.path.exists(self.image_yuan_root), f"path '{self.image_yuan_root}' does not exist."
        assert os.path.exists(self.target_root), f"path '{self.target_root}' does not exist."

        image_jin_names = [p for p in os.listdir(self.image_jin_root) if p.endswith(".jpg")]
        image_yuan_names = [p for p in os.listdir(self.image_yuan_root) if p.endswith(".jpg")]
        image_target_names = [p for p in os.listdir(self.target_root) if p.endswith(".jpg")]

        assert len(image_jin_names) > 0, f"not find any images in {self.image_jin_root}."


        self.images_jin_path = [os.path.join(self.image_jin_root, n) for n in image_jin_names]
        self.images_yuan_path = [os.path.join(self.image_yuan_root, n) for n in image_jin_names]
        self.images_target_path = [os.path.join(self.target_root, n) for n in image_target_names]

        self.transforms = transforms

    def __getitem__(self, idx):
        images_jin_path = self.images_jin_path[idx]
        images_yuan_path = self.images_yuan_path[idx]
        images_target_path = self.images_target_path[idx]

        images_jin = cv2.imread(images_jin_path, flags=cv2.IMREAD_GRAYSCALE)
        images_yuan = cv2.imread(images_yuan_path, flags=cv2.IMREAD_GRAYSCALE)
        target = cv2.imread(images_target_path, flags=cv2.IMREAD_GRAYSCALE)
        assert images_jin is not None, f"failed to read image: {images_jin_path}"
        h, _ = images_jin.shape
        assert target is not None, f"failed to read mask: {images_target_path}"

        if self.transforms is not None:
            image_jin, image_yuan, target = self.transforms(images_jin, images_yuan, target)

        return image_jin, image_yuan, target

    def __len__(self):
        return len(self.images_jin_path)

    @staticmethod
    def collate_fn(batch):

        images_jin, images_yuan, targets = list(zip(*batch))
        batched_imgs_jin = cat_list(images_jin, fill_value=0)
        batched_imgs_yuan = cat_list(images_yuan, fill_value=0)
        batched_targets = cat_list(targets, fill_value=0)

        return batched_imgs_jin, batched_imgs_yuan, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


if __name__ == '__main__':
    train_dataset = DUTSDataset("./", train=True)
    print(len(train_dataset))

    val_dataset = DUTSDataset("./", train=False)
    print(len(val_dataset))

    i, t = train_dataset[0]
