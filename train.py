from losses.yololoss import YoloLoss
from datasets import CustumDataset,VOCDataSets
import torch
import torch.optim as optim
import random
from models.yolo import get_model
import numpy as np
import argparse
import torch.distributed as dist
import json
from torch.nn.parallel import DistributedDataParallel
from utils.model_trainer import ModelTrainer
from config import config
from utils.AverageMeter import AverageMeter
from scf import _single_instance_logger as logger
from scf import setup_single_instance_logger


def train(config):
    rank = config.local_rank
    # root = "/mnt/e/AllData/VOC2012"
    # train_datasets = VOCDataset(True, 640, root)
    # test_datasets = VOCDataset(False, 640, root)

    root = "/mnt/e/AllData/VOC2012"
    data_provider = VOCDataSets(root)
    train_datasets = CustumDataset(augment=True,image_size=640,data_provider=data_provider)
    test_datasets = CustumDataset(augment=False,image_size=640,data_provider=data_provider)

    if config.multi_gpu:
        # DDP模式下，增加采样器
        sampler = torch.utils.data.distributed.DistributedSampler(train_datasets)
        train_loader = torch.utils.data.DataLoader(
            train_datasets,
            batch_size=config.batch_size,
            num_workers=8,
            # shuffle=True,
            sampler=sampler,
            pin_memory=True,
            collate_fn=train_datasets.collate_fn,
        )
    else:
        # 否则没有采样器
        train_loader = torch.utils.data.DataLoader(
            train_datasets, batch_size=config.batch_size, num_workers=8, shuffle=True, pin_memory=True, collate_fn=train_datasets.collate_fn
        )
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(test_datasets, range(100)), batch_size=config.batch_size, num_workers=8, shuffle=True, pin_memory=True, collate_fn=train_datasets.collate_fn
        )
    criterion = YoloLoss(config.num_classes).to(config.device)
    model = get_model(config).to(config.device)
    ModelTrainer.load_model(model,'/home/zhanggong/Extern/workspace/yolo_serise/yolov5_self/jupyter/yolov5_coco.pt')
    optimizer = optim.SGD(model.parameters(), 1e-3, 0.9)

    loss_meter = AverageMeter()

    # DDP时的model改造
    if config.multi_gpu:
        model = DistributedDataParallel(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    for epoch in range(config.epoches):
        ModelTrainer.train(
            data_loader=train_loader,
            model=model,
            loss_meter=loss_meter,
            cur_epoch=epoch,
            cfg=config,
            criterion=criterion,
            optimizer=optimizer,
            logger=logger,
            ids=config.local_rank,
        )
        ModelTrainer.valid(model=model,test_loader=test_loader,head=criterion,epoch=epoch,logger=logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # local_rank 其实是节点的索引
    parser.add_argument("--local_rank", type=int, help="local_rank", default=-1)
    parser.add_argument("--batch_size", type=int, help="总的batch size，每个进程平分", default=4)
    parser.add_argument("--device", type=int, help="device", default=0)  # 单卡下的device
    parser.add_argument("--about", type=str, help="about", default="")  # 设置为特定的标记，用来终止程序
    args = parser.parse_args()

    config.local_rank = args.local_rank
    config.about = args.about

    # batch_size 的定义
    # 每个进程都会跑一个batchsize参数，那么我们怎么定义batch_size这个传入的参数呢？
    # 我们这里定义为总的batch_size
    config.total_batch_size = args.batch_size

    # 认为是ddp模式
    config.multi_gpu = config.local_rank != -1

    # 如果是多卡，也就是DDP模式时
    if config.multi_gpu:
        config.device = f"cuda:{config.local_rank}"
        torch.cuda.set_device(config.device)

        # nccl 是nvidia实现显卡间通信的库，他可以支持例如RingAllReduce，或者nvlink（硬件，显卡高速通信）
        dist.init_process_group(backend="nccl", init_method="env://")

        # 这个是启动的进程数
        config.world_size = dist.get_world_size()
        config.batch_size = config.total_batch_size // config.world_size
    else:
        # 如果是单卡
        config.local_rank = 0  # 0认为是主卡（做log打印，或者储存模型等）
        config.device = f"cuda:{args.device}"
        torch.cuda.set_device(config.device)
        config.world_size = 1
        config.batch_size = config.total_batch_size
    
    setup_single_instance_logger("log")

    # 打印所有的配置信息
    # if config.local_rank == 0: # 说明是主卡
    #     # 配置全局的logger
    #     sys_utils.setup_single_instance_logger("logs/log.txt")
    #     logger.info(f"Startup, config dumps: \n{config}")

    # nn_utils.setup_seed(3)
    train(config)

    # 优雅的收尾
    if config.multi_gpu and config.local_rank == 0:
        dist.destroy_process_group()

    torch.cuda.empty_cache()
