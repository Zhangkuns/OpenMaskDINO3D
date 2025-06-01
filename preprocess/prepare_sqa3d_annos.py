import argparse
import json
import numpy as np
import os
import nltk
import random
from tqdm import tqdm
import torch
from collections import defaultdict
from ChatQformer.utils.box_utils import construct_bbox_corners, box3d_iou
anno_dir = 'annotations/sqa3d'


def convert_person_view(sentence):
    # first-person view to second-person view
    forms = {'i': 'you', 'me': 'you', 'my': 'your', 'mine': 'yours', 'am': 'are'}
    def translate(word):
        if word.lower() in forms:
            return forms[word.lower()]
        return word
    result = ' '.join([translate(word) for word in nltk.wordpunct_tokenize(sentence)])
    return result.capitalize()


def get_sqa_question_type(question):
    question = question.lstrip()
    if question[:4].lower() == 'what':
        return 0
    elif question[:2].lower() == 'is':
        return 1
    elif question[:3].lower() == 'how':
        return 2
    elif question[:3].lower() == 'can':
        return 3
    elif question[:5].lower() == 'which':
        return 4
    else:
        return 5   # others

# 添加参数解析
parser = argparse.ArgumentParser()
parser.add_argument('--segmentor', required=True, type=str)
parser.add_argument('--version', type=str, default='')
parser.add_argument('--train_iou_thres', type=float, default=0.75)
args = parser.parse_args()

segmentor = args.segmentor
version = args.version
train_iou_thres = args.train_iou_thres

for split in ['train', 'val']:
    scan_ids = []
    sqa_annos = []
    question_file = os.path.join(anno_dir, f'v1_balanced_questions_{split}_scannetv2.json')
    with open(question_file, 'r', encoding='utf-8') as f:
        question_data = json.load(f)['questions']
    question_map = {}
    for item in question_data:
        question_map[item['question_id']] = {
            's': [item['situation']] + item['alternative_situation'],   # list of str
            'q': item['question'],   # str
        }

    anno_file = os.path.join(anno_dir, f'v1_balanced_sqa_annotations_{split}_scannetv2.json')
    with open(anno_file, 'r', encoding='utf-8') as f:
        anno_data = json.load(f)['annotations']

    # 加载分割数据
    if segmentor == 'deva':
        seg_gt_ious = torch.load(f"annotations/scannet_{segmentor}_seg_gt_ious.pt", map_location='cpu')
    else:
        instance_attribute_file = f"annotations/scannet_{segmentor}_{split}_attributes{version}.pt"
        scannet_attribute_file = f"annotations/scannet_{split}_attributes.pt"
        instance_attrs = torch.load(instance_attribute_file, map_location='cpu')
        scannet_attrs = torch.load(scannet_attribute_file, map_location='cpu')

    for item in tqdm(anno_data):
        scan_ids.append(item['scene_id'])
        scene_id = item['scene_id']
        obj_id = 0

        # 获取匹配的pred_id
        if segmentor == 'deva':
            if scene_id not in seg_gt_ious:
                continue
            seg_gt_iou = seg_gt_ious[scene_id]
            max_iou, max_id = seg_gt_iou[:, obj_id].max(0)
            max_iou = float(max_iou)
            max_id = int(max_id)
        else:
            if scene_id not in instance_attrs:
                continue
            instance_locs = instance_attrs[scene_id]['locs']
            scannet_locs = scannet_attrs[scene_id]['locs']
            max_iou, max_id = -1, -1
            for pred_id in range(instance_locs.shape[0]):
                pred_locs = instance_locs[pred_id].tolist()
                gt_locs = scannet_locs[obj_id].tolist()
                pred_corners = construct_bbox_corners(pred_locs[:3], pred_locs[3:])
                gt_corners = construct_bbox_corners(gt_locs[:3], gt_locs[3:])
                iou = box3d_iou(pred_corners, gt_corners)
                if iou > max_iou:
                    max_iou = iou
                    max_id = pred_id

        situation = random.choice(question_map[item['question_id']]['s'])
        question = question_map[item['question_id']]['q']
        question_type = get_sqa_question_type(question)
        prompt = situation + ' ' + question + " Answer the question using a single word or phrase."
        answers = [meta['answer'] for meta in item['answers']]

        if split == 'train':
            if max_iou >= args.train_iou_thres:
                answer = random.choice(answers)
                answer = answer.capitalize()
                if answer[-1] != ".":
                    answer += "."
                sqa_annos.append({
                    'scene_id': scene_id,
                    'obj_id': obj_id,
                    'pred_id': max_id,
                    'prompt': prompt,
                    'caption': answer,
                    'sqa_type': question_type,
                    'iou': max_iou
                })
        else:
            sqa_annos.append({
                'scene_id': scene_id,
                'obj_id': obj_id,
                'pred_id': max_id,
                'prompt': prompt,
                'ref_captions': answers,
                'sqa_type': question_type,
                'iou': max_iou
            })

    sqa_annos = sorted(sqa_annos, key=lambda x: (x['scene_id'], x['obj_id']))
    # 保存结果
    with open(f"annotations/sqa3d_{segmentor}_{split}{version}.json", "w") as f:
        json.dump(sqa_annos, f, indent=4)

    

