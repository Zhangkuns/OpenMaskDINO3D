import numpy as np
import json
import sys
sys.path.append('.')
import torch
import random
from tqdm import tqdm
from collections import defaultdict
import argparse
from ChatQformer.utils.box_utils import get_box3d_min_max, box3d_iou, construct_bbox_corners
from ChatQformer.prompts.prompts import grounding_prompt
import string


parser = argparse.ArgumentParser()

parser.add_argument('--segmentor', required=True, type=str)
parser.add_argument('--version', type=str, default='')
parser.add_argument('--train_iou_thres', type=float, default=0.75)
parser.add_argument('--max_obj_num', type=int, default=150)
args = parser.parse_args()

segmentor = args.segmentor
version = args.version

# 遍历训练集和验证集
for split in ["train", "val"]:
    count = [0] * args.max_obj_num  # 初始化计数器
    # 加载ScanRefer数据集注释
    annos = json.load(open(f"annotations/scanrefer/ScanRefer_filtered_{split}.json", "r"))
    # 按场景ID和物体ID排序
    annos = sorted(annos, key=lambda p: f"{p['scene_id']}_{int(p['object_id']):03}")
    new_annos = []

    # 根据不同的分割器加载相应的分割结果
    if segmentor == 'deva':
        seg_gt_ious = torch.load(f"annotations/scannet_{segmentor}_seg_gt_ious.pt", map_location='cpu')
    else:
        # 加载实例属性和场景属性
        instance_attribute_file = f"annotations/scannet_groundtruth_{split}_attributes{version}.pt"
        scannet_attribute_file = f"annotations/scannet_{split}_attributes.pt"
        instance_attrs = torch.load(instance_attribute_file, map_location='cpu')
        scannet_attrs = torch.load(scannet_attribute_file, map_location='cpu')

        # 对于每个标注直接处理：
    for anno in tqdm(annos):
        scene_id = anno['scene_id']
        obj_id = int(anno['object_id'])  # ScanRefer中的object_id
        desc = anno['description']

        # 处理描述文本
        if desc[-1] in string.punctuation:
            desc = desc[:-1]
        prompt = random.choice(grounding_prompt).replace('<description>', desc)

        # 直接使用原始ID
        if split == "train":
            new_annos.append({
                "scene_id": scene_id,
                "obj_id": obj_id,  # 直接使用原始object_id
                "caption": f"<OBJ{obj_id:03}>.",
                "prompt": prompt
            })
        else:
            new_annos.append({
                "scene_id": scene_id,
                "obj_id": obj_id,  # 直接使用原始object_id
                "ref_captions": [f"<OBJ{obj_id:03}>."],
                "prompt": prompt
            })

    print(len(new_annos))

    # 保存结果
    with open(f"annotations/scanrefer_groundtruth_{split}{version}.json", "w") as f:
        json.dump(new_annos, f, indent=4)