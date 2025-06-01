import json
import numpy as np
import sys
import torch
import random
from tqdm import tqdm
from collections import defaultdict
import argparse
from ChatQformer.utils.box_utils import get_box3d_min_max, box3d_iou, construct_bbox_corners

parser = argparse.ArgumentParser()

parser.add_argument('--segmentor', required=True, type=str)
parser.add_argument('--version', type=str, default='')
parser.add_argument('--train_iou_thres', type=float, default=0.75)
args = parser.parse_args()

segmentor = args.segmentor
version = args.version
train_iou_thres = args.train_iou_thres

for split in ['train', 'val']:
    if split == 'test':
        with open(f"annotations/scanqa/ScanQA_v1.0_test_w_obj.json", "r") as f:
            annos = json.load(f)
        with open(f"annotations/scanqa/ScanQA_v1.0_test_wo_obj.json", "r") as f:
            annos.extend(json.load(f))
    else:
        with open(f"annotations/scanqa/ScanQA_v1.0_{split}.json", "r") as f:
            annos = json.load(f)
    print(len(annos))

    new_annos = []
    # 加载分割数据
    if segmentor == 'deva':
        seg_gt_ious = torch.load(f"annotations/scannet_{segmentor}_seg_gt_ious.pt", map_location='cpu')
    else:
        instance_attribute_file = f"annotations/scannet_{segmentor}_{split}_attributes{version}.pt"
        scannet_attribute_file = f"annotations/scannet_{split}_attributes.pt"
        instance_attrs = torch.load(instance_attribute_file, map_location='cpu')
        scannet_attrs = torch.load(scannet_attribute_file, map_location='cpu')

    for anno in tqdm(annos):
        scene_id = anno["scene_id"]
        obj_ids = anno["object_ids"] if "object_ids" in anno else [0]
        question = anno["question"]
        prompt = question + " Answer the question using a single word or phrase."
        answers = anno["answers"] if "answers" in anno else []

        # 获取匹配的pred_id
        if segmentor == 'deva':
            if scene_id not in seg_gt_ious:
                continue
            seg_gt_iou = seg_gt_ious[scene_id]
            # 处理DEVA分割器的情况
            max_iou, max_id = seg_gt_iou[:, obj_ids[0]].max(0)
            max_iou = float(max_iou)
            max_id = int(max_id)
        else:
            # 处理其他分割器(如mask3d)的情况
            if scene_id not in instance_attrs:
                continue
            instance_locs = instance_attrs[scene_id]['locs']
            scannet_locs = scannet_attrs[scene_id]['locs']
            max_iou, max_id = -1, -1
            for pred_id in range(instance_locs.shape[0]):
                pred_locs = instance_locs[pred_id].tolist()
                gt_locs = scannet_locs[obj_ids[0]].tolist()
                pred_corners = construct_bbox_corners(pred_locs[:3], pred_locs[3:])
                gt_corners = construct_bbox_corners(gt_locs[:3], gt_locs[3:])
                iou = box3d_iou(pred_corners, gt_corners)
                if iou > max_iou:
                    max_iou = iou
                    max_id = pred_id

        if split == "train":
            if max_iou >= args.train_iou_thres:
                for i in range(len(answers)):
                    if i > 0 and answers[i] == answers[i - 1]:
                        continue
                    answer = answers[i]
                    answer = answer.capitalize()
                    if answer[-1] != ".":
                        answer += "."
                    new_annos.append({
                        "scene_id": scene_id,
                        "obj_id": obj_ids[0],
                        "pred_id": max_id,
                        "prompt": prompt,
                        "caption": answer,
                        "iou": max_iou
                    })
        else:
            new_annos.append({
                "scene_id": scene_id,
                "obj_id": obj_ids[0],
                "pred_id": max_id,
                "prompt": prompt,
                "ref_captions": answers,
                "iou": max_iou
            })
    print(len(new_annos))
    new_annos = sorted(new_annos, key=lambda x: (x['scene_id'], x['obj_id']))
    with open(f"annotations/scanqa_{segmentor}_{split}{version}.json", "w") as f:
        json.dump(new_annos, f, indent=4)
