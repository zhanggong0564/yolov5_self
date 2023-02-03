import torch
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from comment import nn_utils, sys_utils
from multiprocessing import Lock
import traceback
import json
import cv2
from scf import _single_instance_logger as logger


class Base:
    def __init__(self, cache_file) -> None:
        self.all_labeled_information = []
        self.build_and_cache(cache_file)

    def build_image_and_annotations_generate(self):
        raise NotImplementedError()

    def build_and_cache(self, cache_file):
        if os.path.exists(cache_file):
            logger.info(f"Load labels from cache: {cache_file}")
            self.load_labeled_information_from_cache(cache_file)
        else:
            logger.info(f"Build labels and save to cache: {cache_file}")
            self.build_labeled_information_and_save(cache_file)

    def load_labeled_information_from_cache(self, cache_file):
        """
        从缓存文件中加载标注信息
        """
        self.all_labeled_information, self.label_map = torch.load(cache_file)

    def build_labeled_information_and_save(self, cache_file):
        image_file_and_pixel_annotations_generate, label_map, total_files = self.build_image_and_annotations_generate()
        pbar = tqdm(image_file_and_pixel_annotations_generate, total=total_files, desc="检索图像信息中")
        thread_pool = ThreadPoolExecutor(max_workers=64, thread_name_prefix="prefix_")
        miss_file_log = None
        miss_file_lock = Lock()
        miss_file_json = []
        miss_files = 0

        def process_file(jpeg_file, pixel_annotations):
            nonlocal miss_files, miss_file_log, miss_file_lock
            try:
                pil_image = Image.open(jpeg_file)
                pil_image.verify()
                image_width, image_height = sys_utils.exif_size(pil_image)
                assert image_width > 9 and image_height > 9, f"Image size is too small{image_width} x {image_height}"
            except Exception as e:
                if miss_file_log is None:
                    miss_file_log = open(f"{cache_file}.miss.log", "w")
                miss_file_json.append([jpeg_file, repr(e)])
                miss_file_lock.acquire()
                miss_file_log.write(traceback.format_exc())
                miss_file_log.flush()
                miss_files += 1
                miss_file_lock.release()
                return None
            normalize_annotations = nn_utils.convert_to_normalize_annotation(pixel_annotations, image_width, image_height)
            return [jpeg_file, normalize_annotations, [image_width, image_height]]

        result_futures = []
        for jpeg_file, pixel_annotations in pbar:
            result_futures.append(thread_pool.submit(process_file, jpeg_file, pixel_annotations))
            pbar.set_description(f"Search and cache, total = {total_files}, miss = {miss_files}")
        for item in result_futures:
            result = item.result()
            if result is not None:
                self.all_labeled_information.append(result)
        if miss_file_log is not None:
            miss_file_log.write(json.dumps(miss_file_json, indent=4, ensure_ascii=False) + "\n")
            miss_file_log.close()
        self.label_map = label_map
        sys_utils.mkparents(cache_file)
        torch.save([self.all_labeled_information, self.label_map], cache_file)

    @property
    def num_classes(self):
        return len(self.label_map)

    def __len__(self):
        return len(self.all_labeled_information)

    def __getitem__(self, image_indice):
        jpeg_file, normalize_annotations, (image_width, image_height) = self.all_labeled_information[image_indice]
        image = cv2.imread(jpeg_file)
        return image, normalize_annotations, (image_width, image_height)
