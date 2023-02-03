import torch
import torch.nn as nn
import torchvision


class YoloLoss(torch.nn.Module):
    def __init__(self, num_classes):
        super(YoloLoss, self).__init__()
        self.num_classes = num_classes
        self.anchors = torch.tensor([[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]).view(3, 3, 2)
        self.strides = [8, 16, 32]
        self.offset_boundary = torch.FloatTensor([[+1, 0], [0, +1], [-1, 0], [0, -1]])
        self.num_anchor_per_level = self.anchors.size(1)
        self.num_classes = num_classes
        self.anchor_t = 4.0
        self.BCEClassification = nn.BCEWithLogitsLoss(reduction="mean")
        self.BCEObjectness = nn.BCEWithLogitsLoss(reduction="mean")
        self.balance = [4.0, 1.0, 0.4]  #  8, 16, 32
        self.box_weight = 0.05
        self.objectness_weight = 1.0
        self.classification_weight = 0.5 * self.num_classes / 80  # 80指coco的类别数

    def to(self, device):
        self.anchors = self.anchors.to(device)
        self.offset_boundary = self.offset_boundary.to(device)
        return super().to(device)

    def forward(self, predict, targets):
        """
        predict:(n,75,20,20
                n,75,40,40
                n,75,80,80)

        target: [Nx6] ->[image_id,class_index,cx,cy,width,height]

         # 1.宽宽比,高高比,取最大值,小于阈值anchor_t,被认为是选中  √
                # 2.拓展样本
                # 3.计算GIoU
                # 4.计算loss
                # 5.loss加权合并
        """
        num_target = len(targets)
        device = targets.device
        loss_box_regression = torch.FloatTensor([0]).to(device)
        loss_classification = torch.FloatTensor([0]).to(device)
        loss_objectness = torch.FloatTensor([0]).to(device)
        for i, layer in enumerate(predict):
            layer_h, layer_w = layer.shape[-2:]
            layer = layer.view(-1, 3, 5 + self.num_classes, layer_h, layer_w)
            layer = layer.permute(0, 1, 3, 4, 2).contiguous()
            feature_size_gain = targets.new_tensor([1, 1, layer_w, layer_h, layer_w, layer_h])

            targets_feature_scale = targets * feature_size_gain

            anchors = self.anchors[i]
            num_anchor = len(anchors)
            # 1.计算anchor 和 ground_trues之间的宽高比，小于4.0 的anchor 作为对应的计算

            anchors_wh = anchors.view(num_anchor, 1, 2)
            targets_wh = targets_feature_scale[:, [4, 5]].view(1, num_target, 2)

            wh_ratio = targets_wh / anchors_wh  # num_anchor num_target 2

            max_wh_ratio_values, _ = torch.max(wh_ratio, 1 / wh_ratio).max(dim=2)  # num_anchor num_target
            select_mask = max_wh_ratio_values < self.anchor_t

            # 一个目标需要num_anchor 来负责训练，因此需要复制三份，便于索引targets_feature_scale->nx6->3xnx6
            select_targets = targets_feature_scale.repeat(num_anchor, 1, 1)[select_mask]
            matched_num_target = len(select_targets)
            featuremap_objectness = layer[..., 4]
            objectness_ground_truth = torch.zeros_like(featuremap_objectness)
            if matched_num_target > 0:
                select_anchor_index = torch.arange(num_anchor, device=device).view(num_anchor, 1).repeat(1, num_target)[select_mask]
                # 2. 扩展样本
                select_targets_xy = select_targets[:, [2, 3]]
                xy_divided_one_remainder = select_targets_xy % 1.0

                coord_cell_middle = 0.5
                feature_map_low_boundary = 1.0
                feature_map_high_boundary = feature_size_gain[[2, 3]] - 1.0

                less_x_matched, less_y_matched = ((xy_divided_one_remainder < coord_cell_middle) & (select_targets_xy > feature_map_low_boundary)).T

                greater_x_matched, greater_y_matched = (
                    (xy_divided_one_remainder > (1 - coord_cell_middle)) & (select_targets_xy < feature_map_high_boundary)
                ).T

                select_anchor_index = torch.cat(
                    [
                        select_anchor_index,
                        select_anchor_index[less_x_matched],  # 左边
                        select_anchor_index[less_y_matched],  # 上边
                        select_anchor_index[greater_x_matched],  # 右边
                        select_anchor_index[greater_y_matched],  # 下边
                    ],
                    dim=0,
                )

                select_targets = torch.cat(
                    [
                        select_targets,
                        select_targets[less_x_matched],  # 左边
                        select_targets[less_y_matched],  # 上边
                        select_targets[greater_x_matched],  # 右边
                        select_targets[greater_y_matched],  # 下边
                    ],
                    dim=0,
                )

                xy_offset = torch.zeros_like(select_targets_xy)
                xy_offset = (
                    torch.cat(
                        [
                            xy_offset,
                            xy_offset[less_x_matched] + self.offset_boundary[0],  # 左边
                            xy_offset[less_y_matched] + self.offset_boundary[1],  # 上边
                            xy_offset[greater_x_matched] + self.offset_boundary[2],  # 右边
                            xy_offset[greater_y_matched] + self.offset_boundary[3],  # 下边
                        ]
                    )
                    * coord_cell_middle
                )

                matched_extend_num_target = len(select_targets)
                gt_image_id, gt_class_id = select_targets[:, [0, 1]].long().T
                gt_xy = select_targets[:, [2, 3]]
                gt_wh = select_targets[:, [4, 5]]
                grid_xy = (gt_xy - xy_offset).long()
                grid_x, grid_y = grid_xy.T

                gt_xy = gt_xy - grid_xy

                select_anchors = anchors[select_anchor_index]
                object_predict = layer[gt_image_id, select_anchor_index, grid_y, grid_x]
                object_predict_xy = object_predict[:, [0, 1]].sigmoid() * 2.0 - 0.5
                object_predict_wh = torch.pow(object_predict[:, [2, 3]].sigmoid() * 2.0, 2.0) * select_anchors

                object_predict_box = torch.cat((object_predict_xy, object_predict_wh), dim=1)
                object_ground_truth_box = torch.cat((gt_xy, gt_wh), dim=1)

                gious = self.giou(object_predict_box, object_ground_truth_box)
                giou_loss = 1.0 - gious
                loss_box_regression += giou_loss.mean()

                objectness_ground_truth[gt_image_id, select_anchor_index, grid_y, grid_x] = gious.detach().clamp(0)

                if self.num_classes > 1:
                    object_classification = object_predict[:, 5:]
                    classification_targets = torch.zeros_like(object_classification)
                    classification_targets[torch.arange(matched_extend_num_target), gt_class_id] = 1.0
                    loss_classification += self.BCEClassification(object_classification, classification_targets)
            loss_objectness += self.BCEObjectness(featuremap_objectness, objectness_ground_truth) * self.balance[i]
        num_level = len(predict)
        scale = 3 / num_level
        batch_size = predict[0].shape[0]
        loss_box_regression *= self.box_weight * scale
        loss_objectness *= self.objectness_weight * scale  # 如果 num_level == 4 这里需要乘以1.4，否则乘以1.0
        loss_classification *= self.classification_weight * scale
        loss = loss_box_regression + loss_objectness + loss_classification
        return loss * batch_size

    def giou(self, a, b):
        """
        计算a与b的GIoU
        参数：
        a[Nx4]：      要求是[cx, cy, width, height]
        b[Nx4]:       要求是[cx, cy, width, height]
        GIoU的计算，left = cx - (width - 1) / 2，或者是left = cx - width / 2。两者皆可行
            - 但是，前者的计算与后者在特定场合下，会存在浮点数精度问题。导致小数点后7位不同
            - 如果严格复现，请按照官方写法
            - 如果自己实现，可以选择一种即可
        """
        # a is n x 4
        # b is n x 4

        # cx, cy, width, height
        a_xmin, a_xmax = a[:, 0] - a[:, 2] / 2, a[:, 0] + a[:, 2] / 2
        a_ymin, a_ymax = a[:, 1] - a[:, 3] / 2, a[:, 1] + a[:, 3] / 2
        b_xmin, b_xmax = b[:, 0] - b[:, 2] / 2, b[:, 0] + b[:, 2] / 2
        b_ymin, b_ymax = b[:, 1] - b[:, 3] / 2, b[:, 1] + b[:, 3] / 2

        inter_xmin = torch.max(a_xmin, b_xmin)
        inter_xmax = torch.min(a_xmax, b_xmax)
        inter_ymin = torch.max(a_ymin, b_ymin)
        inter_ymax = torch.min(a_ymax, b_ymax)
        inter_width = (inter_xmax - inter_xmin).clamp(0)
        inter_height = (inter_ymax - inter_ymin).clamp(0)
        inter_area = inter_width * inter_height

        a_width, a_height = (a_xmax - a_xmin), (a_ymax - a_ymin)
        b_width, b_height = (b_xmax - b_xmin), (b_ymax - b_ymin)
        union = (a_width * a_height) + (b_width * b_height) - inter_area
        iou = inter_area / union

        # smallest enclosing box
        convex_width = torch.max(a_xmax, b_xmax) - torch.min(a_xmin, b_xmin) + 1e-16
        convex_height = torch.max(a_ymax, b_ymax) - torch.min(a_ymin, b_ymin)
        convex_area = convex_width * convex_height + 1e-16
        return iou - (convex_area - union) / convex_area
    def detect(self, predict, confidence_threshold=0.3, nms_threshold=0.5, multi_table=True):
        '''
        检测目标
        参数：
        predict[layer8, layer16, layer32],      每个layer是BxCxHxW
        confidence_threshold，                  保留的置信度阈值
        nms_threshold，                         nms的阈值
        '''
        batch = predict[0].shape[0]
        device = predict[0].device
        objs = []
        for ilayer, (layer, stride) in enumerate(zip(predict, self.strides)):
            layer_height, layer_width = layer.size(-2), layer.size(-1)
            layer = layer.view(batch, 3, 5 + self.num_classes, layer_height, layer_width).permute(0, 1, 3, 4, 2).contiguous()
            layer = layer.sigmoid().view(batch, 3, -1, layer.size(-1))
            
            if self.num_classes == 1:
                object_score = layer[..., 4]
                object_classes = torch.zeros_like(object_score)
                keep_batch_indices, keep_anchor_indices, keep_cell_indices = torch.where(object_score > confidence_threshold)
            else:
                layer_confidence = layer[..., [4]] * layer[..., 5:]
                if multi_table:
                    keep_batch_indices, keep_anchor_indices, keep_cell_indices, object_classes = torch.where(layer_confidence > confidence_threshold)
                    object_score = layer_confidence[keep_batch_indices, keep_anchor_indices, keep_cell_indices, object_classes]
                else:
                    object_score, object_classes = layer_confidence.max(-1)
                    keep_batch_indices, keep_anchor_indices, keep_cell_indices = torch.where(object_score > confidence_threshold)
            
            num_keep_box = len(keep_batch_indices)
            if num_keep_box == 0:
                continue

            keepbox = layer[keep_batch_indices, keep_anchor_indices, keep_cell_indices].float()
            layer_anchors = self.anchors[ilayer]
            keep_anchors = layer_anchors[keep_anchor_indices]
            cell_x = keep_cell_indices % layer_width
            cell_y = keep_cell_indices // layer_width
            keep_cell_xy = torch.cat([cell_x.view(-1, 1), cell_y.view(-1, 1)], dim=1)
            wh_restore = (torch.pow(keepbox[:, 2:4] * 2, 2) * keep_anchors) * stride
            xy_restore = (keepbox[:, :2] * 2.0 - 0.5 + keep_cell_xy) * stride
            object_score = object_score.float().view(-1, 1)
            object_classes = object_classes.float().view(-1, 1)
            keep_batch_indices = keep_batch_indices.float().view(-1, 1)
            box = torch.cat((keep_batch_indices, xy_restore - (wh_restore - 1) * 0.5, xy_restore + (wh_restore - 1) * 0.5, object_score, object_classes), dim=1)
            objs.append(box)

        if len(objs) > 0:
            objs_cat = torch.cat(objs, dim=0)
            objs_image_base = []
            for ibatch in range(batch):
                # left, top, right, bottom, score, classes
                select_box = objs_cat[objs_cat[:, 0] == ibatch, 1:]
                objs_image_base.append(select_box)
        else:
            objs_image_base = [torch.zeros((0, 6), device=device) for _ in range(batch)]
        
        if nms_threshold is not None:
            # 使用类内的nms，类间不做操作
            for ibatch in range(batch):
                image_objs = objs_image_base[ibatch]
                if len(image_objs) > 0:
                    max_wh_size = 4096
                    classes = image_objs[:, [5]]
                    bboxes = image_objs[:, :4] + (classes * max_wh_size)
                    confidence = image_objs[:, 4]
                    keep_index = torchvision.ops.boxes.nms(bboxes, confidence, nms_threshold)
                    objs_image_base[ibatch] = image_objs[keep_index]
        return objs_image_base
