import torch
import json
import os
import glob
import numpy as np
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('--scan_dir', required=True, type=str,
                    help='the path of the directory to be saved preprocessed scans')
parser.add_argument('--segmentor', required=True, type=str)
parser.add_argument('--max_inst_num', required=True, type=int)
parser.add_argument('--version', type=str, default='')
args = parser.parse_args()


# 处理训练集和验证集
for split in ["train", "val"]:
    scan_dir = os.path.join(args.scan_dir, 'pcd_all')
    output_dir = "annotations"
    split_path = f"annotations/scannet/scannetv2_{split}.txt"
    # 读取扫描ID列表：
    scan_ids = [line.strip() for line in open(split_path).readlines()]
    scan_ids = sorted(scan_ids)
    print(scan_ids)

    # 读取标签映射文件
    label_map = {}
    with open('/workspace/data/scannet/scannetv2-labels.combined.tsv', 'r') as f:
        # 跳过表头
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            # 将nyu40id作为键，nyuClass作为值存储
            if len(parts) >= 6:  # 确保行有足够的列
                nyu40id = parts[4]  # E列的nyu40id
                nyuclass = parts[6]  # G列的nyuClass
                label_map[nyu40id] = nyuclass

    scans = {}
    # 处理每个扫描
    for scan_id in tqdm(scan_ids):
        pcd_path = os.path.join(scan_dir, f"{scan_id}.pth")
        if not os.path.exists(pcd_path):
            print('skip', scan_id)
            continue
        coords, colors, superpoint, sem_labels, instance_labels, mask3d_instance_class_labels, mask3d_instance_segids = torch.load(pcd_path)

        unique_inst_ids = np.unique(instance_labels)
        unique_inst_ids = unique_inst_ids[unique_inst_ids >= 0]

        inst_locs = []  # 存储所有实例的位置信息
        instance_class_labels = []  # 存储实例的类别标签
        obj_ids = []  # 存储实例的ID
        # 处理每个实例
        for inst_id in unique_inst_ids:
            inst_mask = instance_labels == inst_id  # 获取实例掩码
            pc = coords[inst_mask]  # 获取该实例的点云

            if len(pc) == 0:
                print(scan_id, inst_id.item(), 'empty bbox')
                continue

            obj_ids.append(inst_id)

            # 计算边界框
            size = pc.max(0) - pc.min(0)
            center = (pc.max(0) + pc.min(0)) / 2
            inst_locs.append(np.concatenate([center, size], 0))

            # 获取该实例的语义标签（取众数）
            inst_sem_labels = sem_labels[inst_mask]
            # 使用bincount找出出现最多的标签
            inst_sem_category_id = np.bincount(inst_sem_labels.astype(int)).argmax()
            inst_sem_category = label_map[str(inst_sem_category_id)]
            instance_class_labels.append(inst_sem_category)

        if len(inst_locs) > 0:
            inst_locs = torch.tensor(np.stack(inst_locs, 0), dtype=torch.float32)
            scans[scan_id] = {
                'objects': instance_class_labels,  # (n_obj, )
                'locs': inst_locs,  # (n_obj, 6) center xyz, whl
                'obj_ids': obj_ids  # (n_obj, )
            }

    torch.save(scans, os.path.join(output_dir, f"scannet_groundtruth_{split}_attributes{args.version}.pt"))