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


def read_aggre(name):
    # 讀取聚合JSON文件,返回segment到id的映射
    file = open(name, 'r')
    results = {}
    d = json.load(file)
    l = d['segGroups']
    for i in l:
        for s in i['segments']:
            results[s] = i['id']
    return results

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
    coords = np.ascontiguousarray(points[:, :3]) # 提取xyz坐标

    # 全局对齐变换（可选）：
    if apply_global_alignment:
        align_matrix = np.eye(4)
        with open(os.path.join(scan_dir, scan_id, '%s.txt'%(scan_id)), 'r') as f:
            for line in f:
                if line.startswith('axisAlignment'):
                    align_matrix = np.array([float(x) for x in line.strip().split()[-16:]]).astype(np.float32).reshape(4, 4)
                    break
        # Transform the points
        # 应用变换
        pts = np.ones((coords.shape[0], 4), dtype=coords.dtype)
        pts[:, 0:3] = coords
        coords = np.dot(pts, align_matrix.transpose())[:, :3]  # Nx4
        # Make sure no nans are introduced after conversion
        assert (np.sum(np.isnan(coords)) == 0)

    coords = np.ascontiguousarray(coords - coords.mean(0))
    colors = np.ascontiguousarray(points[:, 3:6]) / 127.5 - 1

    with open(os.path.join(scan_dir, scan_id, '%s_vh_clean_2.labels.ply'%(scan_id)), 'rb') as f2:
        plydata2 = PlyData.read(f2)
    # Map relevant classes to {0,1,...,19}, and ignored classes to -100
    # 將選定的類別映射到0-19, 其他類別映射到-100
    remapper = np.ones(150) * (-100)
    for i, x in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]):
        remapper[x] = i
    sem_labels = np.array(plydata2.elements[0]['label']) # 语义标签
    sem_labels_processed = remapper[np.array(plydata2.elements[0]['label'])] # 语义标签(处理后的)


    fn3 = os.path.join(scan_dir, scan_id, '%s_vh_clean_2.0.010000.segs.json' % (scan_id))
    fn4 = os.path.join(scan_dir, scan_id, '%s.aggregation.json' % (scan_id))
    aggregation = read_aggre(fn4)
    instance_labels = read_segs(fn3, aggregation)

    fn=os.path.join(scan_dir, scan_id, '%s_vh_clean_2.ply'%(scan_id))
    mesh = o3d.io.read_triangle_mesh(fn)
    vertices = torch.from_numpy(np.array(mesh.vertices).astype(np.float32))
    faces = torch.from_numpy(np.array(mesh.triangles).astype(np.int64))
    superpoint = segmentator.segment_mesh(vertices, faces).numpy()

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

    torch.save(
        (coords, colors, superpoint, sem_labels, instance_labels, mask3d_instance_class_labels, mask3d_instance_segids),
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

