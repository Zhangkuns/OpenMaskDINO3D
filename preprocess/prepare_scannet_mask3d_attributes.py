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

    scans = {}
    # 处理每个扫描
    for scan_id in tqdm(scan_ids):
        pcd_path = os.path.join(scan_dir, f"{scan_id}.pth")
        if not os.path.exists(pcd_path):
            print('skip', scan_id)
            continue
        # points, colors, instance_class_labels, instance_segids = torch.load(pcd_path)
        # (coords, colors, superpoint, sem_labels, instance_labels, mask3d_instance_class_labels,
        #  mask3d_instance_segids, superpoint_centers, superpoint_features, instance_sp_masks,
        #  instance_features, num_superpoints, coords_align) = torch.load(pcd_path)
        (coords, coords_align, colors, sem_labels, instance_labels, superpoint,
         superpoint_centers, superpoint_features, num_superpoints, superpoint_semantic_labels,
         mask3d_instance_class_labels, mask3d_instance_segids, mask3d_instance_sp_masks, mask3d_instance_features,
         gt_instance_labels, gt_segids
         ) = torch.load(pcd_path)
        inst_locs = [] # 存储所有实例的位置信息
        num_insts = len(mask3d_instance_class_labels)
        for i in range(min(num_insts, args.max_inst_num)):
            inst_mask = mask3d_instance_segids[i] # 获取实例掩码
            pc = coords_align[inst_mask] # 获取该实例的点云
            if len(pc) == 0: # 如果实例为空
                print(scan_id, i, 'empty bbox')
                inst_locs.append(np.zeros(6, ).astype(np.float32))
                continue
            size = pc.max(0) - pc.min(0)
            center = (pc.max(0) + pc.min(0)) / 2
            inst_locs.append(np.concatenate([center, size], 0))
        inst_locs = torch.tensor(np.stack(inst_locs, 0), dtype=torch.float32)
        scans[scan_id] = {
            'objects': mask3d_instance_class_labels,  # (n_obj, )
            'locs': inst_locs,  # (n_obj, 6) center xyz, whl
        }

    torch.save(scans, os.path.join(output_dir, f"scannet_{args.segmentor}_{split}_attributes{args.version}.pt"))