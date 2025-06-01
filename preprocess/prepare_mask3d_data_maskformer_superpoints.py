import os
import pprint
import time
import multiprocessing as mp
from functools import partial
from plyfile import PlyData
import argparse
import json
import numpy as np
import plyfile
import torch
import csv
import glob
from collections import defaultdict
from tqdm import tqdm
import segmentator
import mmengine
import open3d as o3d
from collections import Counter

def process_instances_with_existing_superpoints(coords, colors, superpoints, mask3d_instance_segids,
                                                mask3d_instance_class_labels):
    """
    使用已有的superpoints处理实例数据

    参数:
        coords: (N, 3) 点云坐标
        colors: (N, 3) 点云颜色
        superpoints: (N,) 每个点所属的superpoint ID
        mask3d_instance_segids: 原始的实例分割点索引列表
        mask3d_instance_class_labels: 实例类别标签
    """
    # 首先计算每个superpoint的特征
    unique_sp_ids = np.unique(superpoints)
    num_superpoints = len(unique_sp_ids)

    # 创建点到superpoint的映射字典
    point_to_sp = {}
    for point_idx, sp_id in enumerate(superpoints):
        point_to_sp[point_idx] = sp_id

    # 计算每个superpoint的特征
    sp_features = []
    sp_centers = []

    for sp_id in unique_sp_ids:
        # 获取属于该superpoint的所有点
        sp_mask = (superpoints == sp_id)
        sp_points = coords[sp_mask]
        sp_colors = colors[sp_mask]

        # 计算特征
        center = sp_points.mean(0)
        std = sp_points.std(0)
        color_mean = sp_colors.mean(0)
        color_std = sp_colors.std(0)

        # 组合特征
        features = np.concatenate([
            center,  # 中心点 (3)
            std,  # 空间分布 (3)
            color_mean,  # 颜色均值 (3)
            color_std,  # 颜色方差 (3)
            np.array([len(sp_points)])  # 点数量 (1)
        ])

        sp_centers.append(center)
        sp_features.append(features)

    # 将特征转换为张量
    sp_features = torch.tensor(np.stack(sp_features), dtype=torch.float32)
    sp_centers = torch.tensor(np.stack(sp_centers), dtype=torch.float32)

    # 处理每个实例，将点级别的mask转换为superpoint级别的mask
    instance_sp_masks = []
    instance_features = []

    for inst_points in mask3d_instance_segids:
        # 找出该实例包含的所有superpoint
        inst_sp_counts = Counter()
        for point_idx in inst_points:
            sp_id = point_to_sp[point_idx]
            inst_sp_counts[sp_id] += 1

        # 创建superpoint mask
        sp_mask = torch.zeros(num_superpoints, dtype=torch.bool)
        for sp_id in inst_sp_counts:
            sp_idx = (unique_sp_ids == sp_id).nonzero()[0]
            sp_mask[sp_idx] = True

        # 计算实例级特征
        inst_features = sp_features[sp_mask].mean(0)

        instance_sp_masks.append(sp_mask)
        instance_features.append(inst_features)

    superpoint_centers = sp_centers
    superpoint_features = sp_features
    instance_sp_masks = torch.stack(instance_sp_masks) if instance_sp_masks else torch.zeros(0)
    instance_features = torch.stack(instance_features) if instance_features else torch.zeros(0)
    num_superpoints = num_superpoints
    return superpoint_centers, superpoint_features, instance_sp_masks, instance_features, num_superpoints


def read_aggre(name):
    # 讀取聚合JSON文件,返回segment到id的映射
    file = open(name, 'r')
    results = {}
    d = json.load(file)
    l = d['segGroups']
    gt_segids = []
    gt_labels = []
    for i in l:
        gt_segids.append(i['segments'])
        gt_labels.append(i['label'])
    for i in l:
        for s in i['segments']:
            results[s] = i['id']
    return results, gt_segids, gt_labels

def read_segs(name, aggregation):
    # 讀取分割數據,將segment映射到實例id
    file = open(name, 'r')
    d = json.load(file)
    indices = np.array(d['segIndices'])
    results = np.zeros_like(indices) - 1
    for i in aggregation:
        m = indices == i
        results[m] = aggregation[i]
    return results


def process_per_scan(scan_id, scan_dir, out_dir, tmp_dir, apply_global_alignment=True, is_test=False):
    pcd_out_dir = os.path.join(out_dir, 'pcd_all')
    print(f"processing {scan_id}...")
    os.makedirs(pcd_out_dir, exist_ok=True)

    # Load point clouds with colors
    # 点云数据加载和处理
    with open(os.path.join(scan_dir, scan_id, '%s_vh_clean_2.ply'%(scan_id)), 'rb') as f:
        plydata = PlyData.read(f) # elements: vertex, fac
    points = np.array([list(x) for x in plydata.elements[0]]) # [[x, y, z, r, g, b, alpha]]
    coords = np.ascontiguousarray(points[:, :3])  # 提取xyz坐标
    ############################################
    coords_align = coords
    # 全局对齐变换（可选）：
    if apply_global_alignment:
        align_matrix = np.eye(4)
        with open(os.path.join(scan_dir, scan_id, '%s.txt' % (scan_id)), 'r') as f:
            for line in f:
                if line.startswith('axisAlignment'):
                    align_matrix = np.array([float(x) for x in line.strip().split()[-16:]]).astype(np.float32).reshape(
                        4, 4)
                    break
        # Transform the points
        # 应用变换
        pts = np.ones((coords.shape[0], 4), dtype=coords.dtype)
        pts[:, 0:3] = coords
        coords_align = np.dot(pts, align_matrix.transpose())[:, :3]  # Nx4
        # Make sure no nans are introduced after conversion
        assert (np.sum(np.isnan(coords_align)) == 0)

    # coords_align = np.ascontiguousarray(coords_align - coords_align.mean(0))
    coords_align = np.ascontiguousarray(coords_align)
    ############################################
    coords = np.ascontiguousarray(points[:, :3] - points[:, :3].mean(0))
    colors = np.ascontiguousarray(points[:, 3:6]) / 127.5 - 1

    with open(os.path.join(scan_dir, scan_id, '%s_vh_clean_2.labels.ply'%(scan_id)), 'rb') as f2:
        plydata2 = PlyData.read(f2)
    # Map relevant classes to {0,1,...,19}, and ignored classes to -100
    # 將選定的類別映射到0-19, 其他類別映射到-100
    remapper = np.ones(150) * (-100)
    for i, x in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]):
        remapper[x] = i
    sem_labels = np.array(plydata2.elements[0]['label']) # nyu40语义标签
    sem_labels_processed = remapper[np.array(plydata2.elements[0]['label'])] # 语义标签(处理后的)

    fn3 = os.path.join(scan_dir, scan_id, '%s_vh_clean_2.0.010000.segs.json' % (scan_id))
    fn4 = os.path.join(scan_dir, scan_id, '%s.aggregation.json' % (scan_id))
    aggregation, gt_segids, gt_instance_labels= read_aggre(fn4)
    instance_labels = read_segs(fn3, aggregation)

    fn=os.path.join(scan_dir, scan_id, '%s_vh_clean_2.ply'%(scan_id))
    mesh = o3d.io.read_triangle_mesh(fn)
    vertices = torch.from_numpy(np.array(mesh.vertices).astype(np.float32))
    faces = torch.from_numpy(np.array(mesh.triangles).astype(np.int64))
    superpoint = segmentator.segment_mesh(vertices, faces).numpy()

    # 在处理超点特征之前添加以下代码
    # 计算每个超点的语义标签
    unique_sp_ids = np.unique(superpoint)
    superpoint_semantic_labels_nyu40 = torch.full((len(unique_sp_ids),), -100, dtype=torch.int64)  # 默认为-100（忽略）
    superpoint_semantic_labels_scannet20 = torch.full((len(unique_sp_ids),), -100, dtype=torch.int64)
    # 将numpy数组转换为torch张量
    superpoint_tensor = torch.from_numpy(superpoint)
    sem_labels_tensor = torch.from_numpy(sem_labels.astype(np.int64))
    sem_labels_processed_tensor = torch.from_numpy(sem_labels_processed.astype(np.int64))

    # 遍历每个超点
    for i, sp_id in enumerate(unique_sp_ids):
        # 获取该超点包含的所有点
        sp_mask_sp_id = (superpoint_tensor == sp_id)
        # 获取这些点的语义标签
        sp_sem_labels = sem_labels_tensor[sp_mask_sp_id]
        sp_sem_labels_scannet20 = sem_labels_processed_tensor[sp_mask_sp_id]
        # 如果有有效标签（不是-100），则取最常见的标签
        valid_labels = sp_sem_labels[sp_sem_labels != -100]
        valid_labels_scannet20 = sp_sem_labels_scannet20[sp_sem_labels_scannet20 != -100]
        if len(valid_labels) > 0:
            # 计算每个标签的频率
            unique_labels, counts = torch.unique(valid_labels, return_counts=True)
            # 选择最常见的标签
            most_common_label = unique_labels[torch.argmax(counts)]
            superpoint_semantic_labels_nyu40[i] = most_common_label
        if len(valid_labels_scannet20) > 0:
            unique_labels, counts = torch.unique(valid_labels_scannet20, return_counts=True)
            most_common_label = unique_labels[torch.argmax(counts)]
            superpoint_semantic_labels_scannet20[i] = most_common_label

    # Map object to segments
    mask3d_instance_class_labels = [] # 实例类别标签
    mask3d_instance_segids = [] # 实例分割标签

    # cur_instances = pointgroup_instances[scan_id].copy()
    # 实例处理
    # 从临时文件加载实例数据，提取标签和分割信息
    if not os.path.exists(os.path.join(tmp_dir, f"{scan_id}.pt")):
        return
    cur_instances = torch.load(os.path.join(tmp_dir, f"{scan_id}.pt"))
    for instance in cur_instances:
        mask3d_instance_class_labels.append(instance["label"])
        mask3d_instance_segids.append(instance["segments"])

    # superpoint_centers, superpoint_features, gt_instance_sp_masks, gt_instance_features, num_superpoints = process_instances_with_existing_superpoints(
    #     coords,
    #     colors,
    #     superpoint,
    #     gt_segids,
    #     gt_instance_labels
    # )

    superpoint_centers, superpoint_features, mask3d_instance_sp_masks,  mask3d_instance_features, num_superpoints = process_instances_with_existing_superpoints(
        coords,
        colors,
        superpoint,
        mask3d_instance_segids,
        mask3d_instance_class_labels
    )


    torch.save(
        (coords,coords_align , colors, sem_labels, instance_labels, superpoint,
         superpoint_centers, superpoint_features, num_superpoints, superpoint_semantic_labels_nyu40, superpoint_semantic_labels_scannet20,
         mask3d_instance_class_labels, mask3d_instance_segids,  mask3d_instance_sp_masks, mask3d_instance_features,
         gt_instance_labels, gt_segids
         ),
        os.path.join(pcd_out_dir, '%s.pth'%(scan_id))
    )
    

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--scannet_dir', required=True, type=str,
                        help='the path to the downloaded ScanNet scans')
    parser.add_argument('--output_dir', required=True, type=str,
                        help='the path of the directory to be saved preprocessed scans')
    parser.add_argument('--class_label_file', required=True, type=str)

    # Optional arguments.
    parser.add_argument('--inst_seg_dir', default=None, type=str)
    parser.add_argument('--segment_dir', default=None, type=str,
                        help='the path to the predicted masks of pretrained segmentor')
    parser.add_argument('--num_workers', default=-1, type=int,
                        help='the number of processes, -1 means use the available max')
    parser.add_argument('--parallel', default=False, action='store_true',
                        help='use mmengine to process in a parallel manner')
    parser.add_argument('--apply_global_alignment', default=False, action='store_true',
                        help='rotate/translate entire scan globally to aligned it with other scans')
    args = parser.parse_args()

    # Print the args
    args_string = pprint.pformat(vars(args))
    print(args_string)

    return args


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    num_workers = args.num_workers

    # 创建类别ID到类别名称的映射字典
    id2class = {}
    with open(args.class_label_file, "r") as f:
        csvreader = csv.reader(f, delimiter='\t')
        csvreader.__next__()
        for line in csvreader:
            id2class[line[0]] = line[2]

    # 如果指定了分割目录
    tmp_dir = args.inst_seg_dir

    # 处理扫描数据
    # for split in ['scans', 'scans_test']:
    for split in ['scans']:
        scannet_dir = os.path.join(args.scannet_dir, split)
        print('scannet_dir:', scannet_dir)
        # 创建处理单个扫描的偏函数
        fn = partial(
            process_per_scan,
            scan_dir=scannet_dir,
            out_dir=args.output_dir,
            tmp_dir=tmp_dir,
            apply_global_alignment=args.apply_global_alignment,
            is_test='test' in split
        )
        # 获取并排序所有扫描ID
        scan_ids = os.listdir(scannet_dir)
        scan_ids.sort()
        print(split, '%d scans' % (len(scan_ids)))

        if args.parallel:
            mmengine.utils.track_parallel_progress(fn, scan_ids, num_workers)
        else:
            for scan_id in scan_ids:
                print(f"processing {scan_id}...")
                fn(scan_id)


if __name__ == '__main__':
    main()

