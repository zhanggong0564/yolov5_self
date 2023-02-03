import numpy as np

class MAPTool:
    def __init__(self,class_names):
        self.class_names = class_names
        self.average_precision_array = np.zeros((len(class_names), ))
        self.map_array = np.zeros((3,))
    def cal_map(self,groundtruth_annotations,detection_annotations,method="interp101"):
        '''

        :param groundtruth_annotations:{image_id: [[left, top, right, bottom, 0, classes_index], [left, top, right, bottom, 0, classes_index]]}
        :param detection_annotations:{image_id: [[left, top, right, bottom, confidence, classes_index], [left, top, right, bottom, confidence, classes_index]]}
        :param method:
        :return:["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
        '''
        average_precision_array = []
        for classes in range(len(self.class_names)):
            matched_table,sum_groundtruth = self.cal_matched_table(groundtruth_annotations,detection_annotations,classes)
            ap_05 = self.compute_average_precision(matched_table, sum_groundtruth, 0.5,method)
            ap_075 = self.compute_average_precision(matched_table, sum_groundtruth, 0.75,method)
            ap_05_095 = np.mean([self.compute_average_precision(matched_table, sum_groundtruth, t,method) for t in np.arange(0.5, 1, 0.05)])
            average_precision_array.append([ap_05, ap_075, ap_05_095])
        self.average_precision_array = average_precision_array
        self.map_array = np.mean(average_precision_array, axis=0)

        names = ["0.5", "0.75", "0.5:0.95"]

        for ap, name in zip(self.map_array, names):
            print(f"Average Precision  (AP) @[ IoU={name:8s} | area=   all | maxDets=100 ] = {ap:.3f}")

        for index, name in enumerate(self.class_names):
            class_ap05, class_ap075, class_ap05095 = self.class_ap(index)
            print(
                f"Class {index:02d}[{name:11s}] mAP@.5 = {class_ap05:.3f},  mAP@.75 = {class_ap075:.3f},  mAP@.5:.95 = {class_ap05095:.3f}")
        return self.map_array
    def class_ap(self, class_name_or_index):
        '''
        # return:
            np.array([ap@0.5, ap@0.75, ap@0.5:0.95])
        '''
        class_index = class_name_or_index
        if isinstance(class_name_or_index, str):
            class_index = self.class_names.index(class_name_or_index)
        return self.average_precision_array[class_index]


    def cal_matched_table(self,groundtruth_annotations,detection_annotations,classes):
        max_dets = 100
        matched_table = []
        sum_groundtruth = 0

        for image_id in groundtruth_annotations:
            select_detection = np.array(
                list(filter(lambda x: x[5] == classes, detection_annotations[image_id])))
            select_groundtruth = np.array(
                list(filter(lambda x: x[5] == classes, groundtruth_annotations[image_id])))

            num_detection = len(select_detection)
            num_groundtruth = len(select_groundtruth)

            num_use_detection = min(num_detection, max_dets)
            sum_groundtruth += num_groundtruth

            if num_detection == 0:
                continue

            if len(select_groundtruth) == 0:
                for index_of_detection in range(num_use_detection):
                    confidence = select_detection[index_of_detection, 4]
                    matched_table.append([confidence, 0, 0, image_id])
                continue

            sgt = select_groundtruth.T.reshape(6, -1, 1)
            sdt = select_detection.T.reshape(6, 1, -1)

            # num_groundtruth x num_detection
            groundtruth_detection_iou = self.iou(sgt, sdt)
            for index_of_detection in range(num_use_detection):
                confidence = select_detection[index_of_detection, 4]
                matched_groundtruth_index = groundtruth_detection_iou[:, index_of_detection].argmax()
                matched_iou = groundtruth_detection_iou[matched_groundtruth_index, index_of_detection]
                matched_table.append([confidence, matched_iou, matched_groundtruth_index, image_id])

        matched_table = sorted(matched_table, key=lambda x: x[0], reverse=True)
        return matched_table,sum_groundtruth
    def compute_average_precision(self,matched_table,sum_groundtruth,ap_threshold,method):
        recall,precision = self.cal_recall_precision(matched_table,sum_groundtruth,ap_threshold)
        average_precision = self.integrate_area_under_curve(precision, recall,method)
        return average_precision


    def cal_recall_precision(self,matched_table,sum_groundtruth,threshold):
        num_dets = len(matched_table)
        true_positive = np.zeros((num_dets,))

        groundtruth_seen_map = {item[3]: set() for item in matched_table}
        for index, (confidence, matched_iou, matched_groundtruth_index, image_id) in enumerate(matched_table):
            image_base_seen_map = groundtruth_seen_map[image_id]
            if matched_iou >= threshold:
                if matched_groundtruth_index not in image_base_seen_map:
                    true_positive[index] = 1
                    image_base_seen_map.add(matched_groundtruth_index)

        num_predicts = np.arange(1, len(true_positive) + 1)
        accumulate_true_positive = np.cumsum(true_positive)
        precision = accumulate_true_positive / num_predicts
        recall = accumulate_true_positive / sum_groundtruth
        return recall,precision
    def integrate_area_under_curve(self,precision, recall,method):
        mrec = np.concatenate(([0.], recall, [min(recall[-1] + 1E-3, 1.)]))
        mpre = np.concatenate(([0.], precision, [0.]))

        # Compute the precision envelope
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

        # Integrate area under curve
        if method == 'interp101':
            x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
            # ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate，梯度积分，https://blog.csdn.net/weixin_44338705/article/details/89203791
            ap = np.mean(np.interp(x, mrec, mpre))  # integrate，直接取均值，论文上都这么做的
        elif method == 'interp11':
            x = np.linspace(0, 1, 11)  # 11-point interp (VOC2007)
            ap = np.mean(np.interp(x, mrec, mpre))  # integrate，直接取均值，论文上都这么做的
        else:  # 'continuous'
            i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes (VOC2012)
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve
        return ap


    def iou(self, a, b):
        aleft, atop, aright, abottom = [a[i] for i in range(4)]
        awidth = aright - aleft + 1
        aheight = abottom - atop + 1

        bleft, btop, bright, bbottom = [b[i] for i in range(4)]
        bwidth = bright - bleft + 1
        bheight = bbottom - btop + 1

        cleft = np.maximum(aleft, bleft)
        ctop = np.maximum(atop, btop)
        cright = np.minimum(aright, bright)
        cbottom = np.minimum(abottom, bbottom)
        cross_area = (cright - cleft + 1).clip(0) * (cbottom - ctop + 1).clip(0)
        union_area = awidth * aheight + bwidth * bheight - cross_area
        return cross_area / union_area




