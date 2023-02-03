import cv2
import numpy as np
import random
import comment.nn_utils as nn_utils


def horizontal_flip(image, normalize_annotations):
    '''
    对图像和框进行水平翻转
    参数：
        image：提供图像
        normalize_annotations：提供归一化后的框信息，格式是[cx, cy, width, height, class_index]
    返回值：
        image, normalize_annotations
    '''
    
    # flipCode = 1 ，   水平，也就是x轴翻转
    # flipCode = 0，    垂直，也就是y轴翻转
    # flipCode = -1，   对角翻转，x和y都发生翻转
    image = cv2.flip(image, flipCode=1)
    #image = np.fliplr(image)
    normalize_annotations = normalize_annotations.copy()

    # cx, cy, width, height
    # 0-1
    # (image_width - 1) / image_width
    image_width = image.shape[1]  # Height, Width, Channel
    normalize_annotations[:, 0] = (image_width - 1) / image_width - normalize_annotations[:, 0]
    return image, normalize_annotations


def hsv_augument(image, hue_gain=0.015, sat_gain=0.7, value_gain=0.4):
    hue_gain = np.random.uniform(-1, 1) * hue_gain + 1
    sat_gain = np.random.uniform(-1, 1) * sat_gain + 1
    value_gain = np.random.uniform(-1, 1) * value_gain + 1

    h, s, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    lut_base = np.arange(0, 256)
    lut_hue = ((lut_base * hue_gain) % 180).astype(image.dtype)
    lut_sat = np.clip(lut_base * sat_gain, 0, 255).astype(image.dtype)
    lut_value = np.clip(lut_base * value_gain, 0, 255).astype(image.dtype)

    changed_hue = cv2.LUT(h, lut_hue)
    changed_sat = cv2.LUT(s, lut_sat)
    changed_value = cv2.LUT(v, lut_value)

    image_hsv = cv2.merge((changed_hue, changed_sat, changed_value))
    image_bgr = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)
    return image_bgr
def load_mosaic(image_idx,image_size,provider,border_fill_tuple):
        x_center = int(random.uniform(image_size * 0.5, image_size * 1.5))
        y_center = int(random.uniform(image_size * 0.5, image_size * 1.5))
        num_image = len(provider)

        all_image_index = [image_idx] + [random.randint(0, num_image - 1) for _ in range(3)]

        alignment_corner_point = [[1, 1], [0, 1], [1, 0], [0, 0]]
        merge_mosaic_image_size = image_size * 2
        merge_mosaic_image = np.full((merge_mosaic_image_size, merge_mosaic_image_size, 3), (114, 114, 114), dtype=np.uint8)
        merge_mosaic_pixel_annotations = []
        for image_index, align_point in zip(all_image_index, alignment_corner_point):
            image, normalize_annotations, (image_width, image_height) = provider[image_index]
            align_point_x, align_point_y = align_point
            align_point_x = align_point_x * image_width
            align_point_y = align_point_y * image_height
            x_offset = x_center - align_point_x
            y_offset = y_center - align_point_y

            M = np.array([[1, 0, x_offset], [0, 1, y_offset]], dtype=np.float32)
            cv2.warpAffine(
                image, M, (merge_mosaic_image_size, merge_mosaic_image_size), dst=merge_mosaic_image, borderMode=cv2.BORDER_TRANSPARENT, flags=cv2.INTER_NEAREST
            )
            pixel_annotations = nn_utils.convert_to_pixel_annotation(normalize_annotations, image_width, image_height)
            pixel_annotations += [x_offset, y_offset, x_offset, y_offset, 0]
            merge_mosaic_pixel_annotations.append(pixel_annotations)

        merge_mosaic_pixel_annotations = np.concatenate(merge_mosaic_pixel_annotations, axis=0)
        np.clip(merge_mosaic_pixel_annotations[:, :4], a_min=0, a_max=merge_mosaic_image_size - 1, out=merge_mosaic_pixel_annotations[:, :4])

        # 随机仿射变换
        scale = random.uniform(0.5, 1.5)
        M = np.array([[scale, 0, image_size * (0.5 - scale)], [0, scale, image_size * (0.5 - scale)]], dtype=np.float32)  # 2*3 x 3*2n
        merge_mosaic_image = cv2.warpAffine(
            merge_mosaic_image,
            M,
            (image_size, image_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_fill_tuple,
        )

        num_targets = len(merge_mosaic_pixel_annotations)
        output_normalize_annotations = np.zeros((0, 5))

        if num_targets > 0:
            targets_temp = np.ones((num_targets * 2, 3))
            targets_temp[:, :2] = merge_mosaic_pixel_annotations[:, :4].reshape(-1, 2)
            merge_projection_pixel_annotations = merge_mosaic_pixel_annotations.copy()
            merge_projection_pixel_annotations[:, :4] = (targets_temp @ M.T).reshape(-1, 4)

            np.clip(merge_projection_pixel_annotations[:, :4], 0, image_size - 1, out=merge_projection_pixel_annotations[:, :4])
            # 保留的条件分析
            # 1. 映射后的框，宽度必须大于2
            # 2. 映射后的框，高度必须大于2
            # 3. 裁切后的面积 / 裁切前的面积 > 0.2
            # 4. max(宽高比，高宽比) < 20
            # coding
            project_box_width = merge_projection_pixel_annotations[:, 2] - merge_projection_pixel_annotations[:, 0] + 1
            project_box_height = merge_projection_pixel_annotations[:, 3] - merge_projection_pixel_annotations[:, 1] + 1

            original_width = merge_mosaic_pixel_annotations[:, 2] - merge_mosaic_pixel_annotations[:, 0] + 1
            original_height = merge_mosaic_pixel_annotations[:, 3] - merge_mosaic_pixel_annotations[:, 1] + 1

            area_aspect_ratio = (project_box_width * project_box_height) / (original_width * original_height)
            wh_aspect_ratio = np.maximum(project_box_width / (project_box_height + 1e-5), project_box_height / (project_box_width + 1e-5))

            keep_index = (project_box_width > 2) & (project_box_height > 2) & (area_aspect_ratio > 0.2) & (wh_aspect_ratio < 20)
            merge_projection_pixel_annotations = merge_projection_pixel_annotations[keep_index]

            output_normalize_annotations = nn_utils.convert_to_normalize_annotation(merge_projection_pixel_annotations, image_size, image_size)

        return merge_mosaic_image, output_normalize_annotations


if __name__ == "__main__":
    image = cv2.imread("/home/zhanggong/Extern/workspace/yolo_serise/yolov5_self/image.jpg")
    image_hsv = hsv_augument(image)

    cv2.imshow("image", image)
    cv2.imshow("image_hsv", image_hsv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
