import torch
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from ChatQformer.dataset.dataloader import MetaLoader
from ChatQformer.dataset.dataset_train import TrainDataset
from ChatQformer.dataset.dataset_val import ValDataset

import logging
logger = logging.getLogger(__name__)


def create_dataset(config):
    # 如果是评估模式
    if config.evaluate:
        train_datasets = []
    else:
        train_files = []  # 初始化训练文件列表
        for train_name in config.train_tag.split('#'):
            if train_name not in config.train_file_dict:
                raise NotImplementedError
            appendfile = config.train_file_dict[train_name]
            train_files.append(appendfile)
        
        train_datasets = []  # 初始化训练数据集列表
        datasets = []  # 临时数据集列表
        for train_file in train_files:  # 遍历训练文件
            datasets.append(TrainDataset(ann_list=train_file, config=config))  # 创建训练数据集
        dataset = ConcatDataset(datasets)  # 合并数据集
        train_datasets.append(dataset)  # 添加到训练数据集列表

    val_files = {}
    for val_name in config.val_tag.split('#'):
        if val_name not in config.val_file_dict:
            raise NotImplementedError
        val_files[val_name] = config.val_file_dict[val_name]

    val_datasets = []  # 初始化验证文件字典
    for k, v in val_files.items():
        datasets = []
        if type(v[0]) != list:
            v = [v]
        for val_file in v:
            datasets.append(ValDataset(ann_list=val_file, dataset_name=k, config=config))
        dataset = ConcatDataset(datasets)
        val_datasets.append(dataset)

    return train_datasets, val_datasets


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []  # 初始化采样器列表
    for dataset, shuffle in zip(datasets, shuffles):  # 遍历数据集和对应的shuffle标志
        sampler = torch.utils.data.DistributedSampler(  # 创建分布式采样器
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle
        )
        samplers.append(sampler)  # 添加到采样器列表
    return samplers


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []  # 初始化加载器列表
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(
        datasets, samplers, batch_size, num_workers, is_trains, collate_fns
    ):
        if is_train:  # 如果是训练模式
            shuffle = sampler is None  # 没有采样器时才shuffle
            drop_last = True  # 丢弃最后不完整的batch
        else:  # 如果是验证模式
            shuffle = False  # 不shuffle
            drop_last = False  # 保留最后不完整的batch
        loader = DataLoader(  # 创建数据加载器
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=False,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
            persistent_workers=True if n_worker > 0 else False,
        )
        loaders.append(loader)  # 添加到加载器列表
    return loaders  # 返回加载器列表


def iterate_dataloaders(dataloaders):
    """Alternatively generate data from multiple dataloaders,
    since we use `zip` to concat multiple dataloaders,
    the loop will end when the smaller dataloader runs out.

    Args:
        dataloaders List(DataLoader): can be a single or multiple dataloaders
    """
    for data_tuples in zip(*dataloaders):  # 同时遍历多个数据加载器
        for idx, data in enumerate(data_tuples):  # 遍历每个batch的数据
            yield dataloaders[idx].dataset.media_type, data  # 返回媒体类型和数据
