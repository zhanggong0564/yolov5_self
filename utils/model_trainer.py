import torch
import numpy as np
from tqdm import tqdm
from collections import Counter
from torch.nn import functional as F
import os
from comment import nn_utils
from evaluate.evaluation import MAPTool


class ModelTrainer(object):
    @staticmethod
    def get_lr(optimizer):
        """Get the current learning rate from optimizer."""
        for param_group in optimizer.param_groups:
            return param_group["lr"]

    @staticmethod
    def fix_bn(m):
        classname = m.__class__.__name__
        if classname.find("BatchNorm") != -1:
            m.eval()

    @staticmethod
    def load_model(model, model_path):
        weights = torch.load(model_path, map_location="cpu")
        model_dict = model.state_dict()
        for k in model_dict:
            if k not in weights or model_dict[k].shape != weights[k].shape:
                print(f"{k} is miss match {model_dict[k].shape}-{weights[k].shape} ")
            else:
                model_dict[k] = weights[k]
        # weights = {k: v for k, v in weights.items() if k in model_dict and model_dict[k].shape == v.shape}
        # model_dict.update(weights)
        model.load_state_dict(model_dict)
        print("load model finish")
        return model

    @staticmethod
    def train(data_loader, model, criterion, loss_meter, optimizer, cur_epoch, cfg, logger, ids=0):
        model.train()
        data_loader = tqdm(data_loader)
        data_loader.set_description(f"Epoch{cur_epoch + 1}")
        for batch_idx, (images, labels, _) in enumerate(data_loader):
            images = images.to(cfg.device)
            labels = labels.to(cfg.device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item(), images.shape[0])
            lr = ModelTrainer.get_lr(optimizer)
            data_loader.set_postfix(lr=lr, loss=loss.item())

            if batch_idx % cfg.print_freq == 0 and batch_idx != 0 and ids == 0:
                loss_avg = loss_meter.avg
                logger.info("Epoch %d, iter %d/%d, lr %f, loss %f" % (cur_epoch, batch_idx, len(data_loader), lr, loss_avg))
                loss_meter.reset()
            if (batch_idx + 1) % cfg.save_freq == 0 and batch_idx != 0 and ids == 0:
                saved_name = "Epoch_%d_batch_%d.pt" % (cur_epoch, batch_idx)
                state = {
                    # "state_dict": model.module.state_dict(),
                    "state_dict": model.state_dict(),
                    "epoch": cur_epoch,
                }
                torch.save(state, os.path.join(cfg.out_dir, saved_name))
                logger.info("Save checkpoint %s to disk." % saved_name)
        if ids == 0:
            saved_name = "Epoch_%d.pt" % cur_epoch
            state = {
                # "state_dict": model.module.state_dict(),
                "state_dict": model.state_dict(),
                "epoch": cur_epoch,
                "optimizer": optimizer.state_dict(),
            }
            torch.save(state, os.path.join(cfg.out_dir, saved_name))
            logger.info("Save checkpoint %s to disk..." % saved_name)
        # return model

    @staticmethod
    def distill_train(data_loader, student_model, teacher_model, criterion1, criterion2, loss_meter, optimizer, cur_epoch, device, cfg, logger):
        student_model.train()
        teacher_model.eval()
        data_loader = tqdm(data_loader)
        data_loader.set_description(f"Epoch{cur_epoch}")
        for batch_idx, (images, labels, _) in enumerate(data_loader):
            images = images.to(device)
            labels = labels.to(device)
            labels = labels.squeeze()
            feat_student, output = student_model.forward(images, labels)
            loss1 = criterion1(output, labels)
            with torch.no_grad():
                feat_teacher = teacher_model.forward(images)

            loss2 = criterion2(feat_student, feat_teacher)
            loss3 = torch.mean(torch.sum((F.normalize(feat_teacher) - F.normalize(feat_student)) ** 2, dim=1))
            loss = loss1 + loss2 + loss3
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item(), images.shape[0])
            loss_avg = loss_meter.avg
            lr = ModelTrainer.get_lr(optimizer)
            data_loader.set_postfix(lr=lr, loss=loss_avg)
            if batch_idx % cfg.print_freq == 0 and batch_idx != 0:
                logger.info("Epoch %d, iter %d/%d, lr %f, loss %f" % (cur_epoch, batch_idx, len(data_loader), lr, loss_avg))
                loss_meter.reset()
            if (batch_idx + 1) % cfg.save_freq == 0:
                saved_name = "Epoch_%d_batch_%d.pt" % (cur_epoch, batch_idx)
                state = {"state_dict": student_model.module.state_dict(), "epoch": cur_epoch, "batch_id": batch_idx}
                torch.save(state, os.path.join(cfg.out_dir, saved_name))
                logger.info("Save checkpoint %s to disk." % saved_name)
            loss_meter.reset()
        saved_name = "Epoch_%d.pt" % cur_epoch
        state = {"state_dict": student_model.module.state_dict(), "epoch": cur_epoch, "optimizer": optimizer.state_dict()}
        torch.save(state, os.path.join(cfg.out_dir, saved_name))
        logger.info("Save checkpoint %s to disk..." % saved_name)

    @staticmethod
    def valid(model, test_loader, head, epoch, logger):
        model.eval()
        param = next(model.parameters())
        device = param.device
        dtype = param.dtype
        batch_size = test_loader.batch_size
        MAP = MAPTool(test_loader.dataset.dataset.label_map)

        with torch.no_grad():

            groundtruth_annotations = {}
            detection_annotations = {}

            # test loader是使用centerAffine进行的
            # normalize_annotations格式是[image_id, class_index, cx, cy, width, height]
            for batch_index, (images, normalize_annotations, visual) in enumerate(tqdm(test_loader, desc=f"Eval map {epoch:03d} epoch")):
                images = images.to(device, non_blocking=True).type(dtype)
                predicts = model(images)

                # 检测目标，得到的结果是[left, top, right, bottom, confidence, classes]
                objects = head.detect(predicts, confidence_threshold=0.001, nms_threshold=0.6)

                batch, channels, image_height, image_width = images.shape
                visual_image_id, visual_image, visual_annotations, restore_info = visual

                num_batch = images.shape[0]
                normalize_annotations = normalize_annotations.to(device)
                restore_info = normalize_annotations.new_tensor(restore_info)  # pad_left, pad_top, origin_width, origin_height, scale

                pixel_annotations = nn_utils.convert_to_pixel_annotation(normalize_annotations[:, [2, 3, 4, 5, 0, 1]], image_width, image_height)
                for i in range(num_batch):
                    index = torch.where(pixel_annotations[:, 4] == i)[0]
                    if len(index) == 0:
                        continue

                    padx, pady, origin_width, origin_height, scale = restore_info[i]
                    pixel_annotations[index, :4] = (pixel_annotations[index, :4] - restore_info[i, [0, 1, 0, 1]]) / scale

                for left, top, right, bottom, image_id, class_id in pixel_annotations.cpu().numpy():
                    image_id = int(image_id) + batch_index * batch_size
                    class_id = int(class_id)
                    if image_id not in groundtruth_annotations:
                        groundtruth_annotations[image_id] = []

                    groundtruth_annotations[image_id].append([left, top, right, bottom, 0, class_id])

                for image_index, image_objs in enumerate(objects):
                    image_objs[:, 0].clamp_(0, image_width)
                    image_objs[:, 1].clamp_(0, image_height)
                    image_objs[:, 2].clamp_(0, image_width)
                    image_objs[:, 3].clamp_(0, image_height)

                    padx, pady, origin_width, origin_height, scale = restore_info[image_index]
                    image_objs[:, :4] = (image_objs[:, :4] - restore_info[image_index, [0, 1, 0, 1]]) / scale
                    image_id = image_index + batch_index * batch_size
                    detection_annotations[image_id] = image_objs.cpu().numpy()

            # merge groundtruth_annotations
            for image_id in groundtruth_annotations:
                groundtruth_annotations[image_id] = np.array(groundtruth_annotations[image_id], dtype=np.float32)
            map_array = MAP.cal_map(groundtruth_annotations, detection_annotations)
            # map_result = maptool.MAPTool(groundtruth_annotations, detection_annotations, test_loader.dataset.provider.label_map)
            map05, map075, map05095 = map_array
            model_score = map05 * 0.1 + map05095 * 0.9
            logger.info(f"Eval {epoch:03d} epoch, mAP@.5 [{map05:.6f}], mAP@.75 [{map075:.6f}], mAP@.5:.95 [{map05095:.6f}]")
        return model_score
