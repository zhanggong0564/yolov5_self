import os
import sys
import cv2
import torch
from torch.utils.data import Dataset
import comment.nn_utils as nn_utils
import comment.sys_utils as sys_utils
from tqdm import tqdm
import random
import numpy as np
import torchvision.transforms.functional as T
from datasets.voc import VOCDataSets
from datasets.augument import load_mosaic, hsv_augument, horizontal_flip


class CustumDataset(Dataset):
    def __init__(self, augment, image_size, data_provider):
        super(CustumDataset, self).__init__()
        self.augment = augment
        self.image_size = image_size
        self.border_fill_value = 114
        self.border_fill_tuple = self.border_fill_value, self.border_fill_value, self.border_fill_value
        self.provider = data_provider

    def __getitem__(self, index):
        if self.augment and random.random() < 0.5:
            restore_info = None
            image, normalize_annotations = load_mosaic(index, self.image_size, self.provider, self.border_fill_tuple)
            image = hsv_augument(image)
            if random.random() < 0.5:
                image, normalize_annotations = horizontal_flip(image, normalize_annotations)
        else:
            image, normalize_annotations, restore_info = self.load_center_affine(index)
        BGR = image
        image = cv2.cvtColor(BGR,cv2.COLOR_BGR2RGB).transpose(2,0,1)
        image =image/ np.array([255.0], dtype=np.float32)
        output_annotations = np.zeros(( len(normalize_annotations), 6), dtype=np.float32)
        output_annotations[:, 1:] = normalize_annotations[:, [4, 0, 1, 2, 3]]
        return torch.from_numpy(image),BGR,torch.from_numpy(output_annotations),normalize_annotations, restore_info

    def __len__(self):
        return len(self.provider)

    def load_image_with_uniform_scale(self, image_index):
        """
        :param image_index:加载图像进行长边的等比缩放
        :return:
        """
        image, normalize_annotations, (w, h) = self.provider[image_index]
        scale = self.image_size / max(image.shape[:2])
        if scale != 1:
            if not self.augment and scale < 1:
                interp = cv2.INTER_AREA  # 速度慢，效果好，区域插值
            else:
                interp = cv2.INTER_LINEAR  # 速度快，效果也还ok，线性插值
            image = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=interp)
        image_resized_height, image_resized_width = image.shape[:2]
        return image, normalize_annotations.copy(), (image_resized_width, image_resized_height), scale

    def load_center_affine(self, index):
        image, normalize_annotations, (width, height), scale = self.load_image_with_uniform_scale(index)
        pad_width = self.image_size - width
        pad_height = self.image_size - height

        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top

        image = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=self.border_fill_tuple)

        x_alpha = width / self.image_size
        x_beta = pad_left / self.image_size

        y_alpha = height / self.image_size
        y_beta = pad_top / self.image_size

        normalize_annotations[:, [0, 1]] = normalize_annotations[:, [0, 1]] * [x_alpha, y_alpha] + [x_beta, y_beta]
        normalize_annotations[:, [2, 3]] = normalize_annotations[:, [2, 3]] * [x_alpha, y_alpha]
        return image, normalize_annotations.copy(), (pad_left, pad_top, width, height, scale)

    @staticmethod
    def collate_fn(batch):
        images, original_images, normalize_annotations, original_normalize_annotations, restore_info = zip(*batch)
        for index, annotations in enumerate(normalize_annotations):
            annotations[:, 0] = index

        # 准备visual_info，用来显示的东西， image, annotations, image_id
        visual_image_id = random.randint(0, len(original_images)-1)
        visual_image = original_images[visual_image_id]
        visual_annotations = original_normalize_annotations[visual_image_id]
        visual_info = visual_image_id, visual_image, visual_annotations, restore_info

        normalize_annotations = torch.cat(normalize_annotations, dim=0)
        images = torch.stack(images, dim=0)
        return images, normalize_annotations, visual_info


# def draw_image(image,normalize_annotations):
#     h,w,c = image.shape
#     nor
if __name__ == "__main__":

    # anno_files = '/mnt/e/AllData/VOC2012/Annotations/2007_000027.xml'
    # label_map = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    #                           "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
    #                           "tvmonitor"]
    root = "/mnt/e/AllData/VOC2012"
    data_provider = VOCDataSets(root)
    datasets = CustumDataset(augment=True, image_size=640, data_provider=data_provider)
    # image, normalize_annotations = datasets[0]
    nn_utils.setup_seed(20)
    _,image,_,merge_projection_pixel_annotations, _ = datasets[100]
    merge_projection_pixel_annotations = nn_utils.convert_to_pixel_annotation(merge_projection_pixel_annotations, 640, 640)
    for left, top, right, bootom, _ in merge_projection_pixel_annotations:
        nn_utils.draw_bbox(image, left, top, right, bootom, 1, 0)
        nn_utils.draw_norm_bboxes

    print(image.shape)
    cv2.imshow("image", image)
    cv2.waitKey()
    cv2.destroyAllWindows()
    # print(datasets[0])
