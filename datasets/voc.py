from comment import sys_utils,nn_utils
from datasets.base import Base
import os
class VOCDataSets(Base):
    '''
    VOC的数据提供者
    '''
    def __init__(self, root, cache_root="dataset_cache"):
        self.root = root

        cache_name = sys_utils.get_md5(root)
        super().__init__(f"{cache_root}/voc_{cache_name}.cache")

    
    def build_image_and_annotations_generate(self):
        # 生成器, label_map，total[用来显示进度的]
        annotations_files = os.listdir(os.path.join(self.root, "Annotations"))

        # 保留所有的xml后缀文件
        annotations_files = list(filter(lambda x: x.endswith(".xml"), annotations_files))
        total_files = len(annotations_files)

        # xml改jpg
        jpeg_files = [item[:-3] + "jpg" for item in annotations_files]
        
        # 把文件名修改为全路径
        annotations_files = map(lambda x: os.path.join(self.root, "Annotations", x), annotations_files)
        jpeg_files = map(lambda x: os.path.join(self.root, "JPEGImages", x), jpeg_files)
        
        label_map = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

        def generate_function():
            for jpeg_file, annotation_file in zip(jpeg_files, annotations_files):
                pixel_annotations = nn_utils.load_voc_annotation(annotation_file, label_map)
                yield jpeg_file, pixel_annotations

        return generate_function(), label_map, total_files
if __name__ == '__main__':
    root = "/mnt/e/AllData/VOC2012"
    datasets = VOCDataSets(root)
    print(datasets[0])