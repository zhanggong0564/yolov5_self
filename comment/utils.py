import  numpy as np


def convert_to_normalize_annotation(pixel_annotations, image_width, image_height):
    normalize_annotations = pixel_annotations.copy()
    left, top, right, bottom, _ = [pixel_annotations[:,i]  for i in range(5)]
    normalize_annotations[:, 0] = (left + right) * 0.5 / image_width  # cx
    normalize_annotations[:, 1] = (top + bottom) * 0.5 / image_height  # cy
    normalize_annotations[:, 2] = (right - left + 1) / image_width  # width
    normalize_annotations[:, 3] = (bottom - top + 1) / image_height  # height
    return normalize_annotations

def convert_to_pixel_annotation(normalize_annotations,image_with,image_height):
    '''
    normalize_annotations:Nx5
    '''
    pixel_annotations = normalize_annotations.copy()
    cx,cy,width,height = [normalize_annotations[:,i] for i in range(4)]
    pixel_annotations[:,0] = cx*image_with-(width*image_with-1)*0.5
    pixel_annotations[:,1] = cy*image_height-(height*image_height-1)*0.5
    pixel_annotations[:,2] = cx*image_with-(width*image_with-1)*0.5
    pixel_annotations[:,3] = cy*image_height-(height*image_height-1)*0.5
    return pixel_annotations


def load_voc_annotation(annotation_file, label_map):
    '''
    加载标注文件xml，读取其中的bboxes
    参数：
        annotation_file[str]：  指定为xml文件路径
        label_map[list]：       指定为标签数组
    返回值：
        np.array([(xmin, ymin, xmax, ymax, class_index), (xmin, ymin, xmax, ymax, class_index)])
    '''
    with open(annotation_file, "r") as f:
        annotation_data = f.read()

    def middle(s, begin, end, pos_begin=0):
        '''
        从开始的位置 到结束的位置，找到中间的内容
        :param s: 待查找的字符串
        :param begin: 开始位置的字符串
        :param end: 结束位置的字符传
        :param pos_begin: 开始查找位置的索引
        :return: 中间的字符串，结束的位置
        '''
        start_index = s.find(begin,pos_begin)
        if start_index==-1:
            return None,None
        start_index = start_index+len(begin)
        end_index = s.find(end,start_index)
        if end_index==-1:
            return None,None

        return s[start_index:end_index], end_index + len(end)

    obj_bboxes =[]
    object_ ,pos_= middle(annotation_data,'<object>','</object>')
    while object_ is not None:
        xmin = int(float(middle(object_, "<xmin>", "</xmin>")[0]))
        ymin = int(float(middle(object_, "<ymin>", "</ymin>")[0]))
        xmax = int(float(middle(object_, "<xmax>", "</xmax>")[0]))
        ymax = int(float(middle(object_, "<ymax>", "</ymax>")[0]))
        name = middle(object_,"<name>", "</name>")[0]
        object_,pos_ = middle(annotation_data, "<object>", "</object>", pos_)
        obj_bboxes.append((xmin, ymin, xmax, ymax, label_map.index(name)))
    return_ndarray_bboxes = np.zeros((0,5),dtype=np.float32)
    if len(obj_bboxes)>0:
        return_ndarray_bboxes = np.array(obj_bboxes,dtype=np.float32)
    return return_ndarray_bboxes
if __name__ == '__main__':

    anno_files = '/mnt/e/AllData/VOC2012/Annotations/2007_000027.xml'
    label_map = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
                              "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
                              "tvmonitor"]

    print(load_voc_annotation(anno_files,label_map))