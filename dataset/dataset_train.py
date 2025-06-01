import logging
import os
import json

import numpy as np
import torch

from ChatQformer.dataset.base_dataset import BaseDataset, update_caption
import glob
import random
from ChatQformer.prompts.prompts import obj_caption_wid_prompt
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)



class TrainDataset(BaseDataset):

    cached_feats = {}

    def __init__(self, ann_list, config, **kwargs):
        super().__init__()
        # 从配置中获取维度和最大物体数量
        self.feat_dim = config.model.input_dim
        self.img_feat_dim = config.model.img_input_dim
        self.max_obj_num = config.model.max_obj_num
        self.pcd_dir = config.model.pcd_dir
        # 解析输入列表中的文件路径
        feat_file, img_feat_file, attribute_file, anno_file = ann_list[:4]

        # 加载属性和标注数据
        self.attributes = torch.load(attribute_file, map_location='cpu') if attribute_file is not None else None
        self.anno = json.load(open(anno_file, 'r'))

        if len(ann_list) > 4:
            sample_ratio = ann_list[-1]
            if sample_ratio < 1:
                self.anno = random.sample(self.anno, int(sample_ratio * len(self.anno)))

        # 检查是否已经缓存了特征
        if feat_file in TrainDataset.cached_feats and img_feat_file in TrainDataset.cached_feats:
            self.scene_feats, self.scene_masks = TrainDataset.cached_feats[feat_file]
            self.scene_img_feats = TrainDataset.cached_feats[img_feat_file]
        else:
            # 如果没有缓存，加载特征文件
            if feat_file is not None and os.path.exists(feat_file):
                self.feats = torch.load(feat_file, map_location='cpu')
            else:
                self.feats = None
            if img_feat_file is not None and os.path.exists(img_feat_file):
                self.img_feats = torch.load(img_feat_file, map_location='cpu')
            else:
                self.img_feats = None
            # 处理场景特征
            if self.attributes is None:
                self.scene_feats = self.feats
                self.scene_img_feats = self.scene_masks = None
            else:
                self.scene_feats, self.scene_img_feats, self.scene_masks = self.prepare_scene_features()
            TrainDataset.cached_feats[feat_file] = (self.scene_feats, self.scene_masks)
            TrainDataset.cached_feats[img_feat_file] = self.scene_img_feats


    def __len__(self):
        return len(self.anno)

    def __getitem__(self, index):
        # 如果场景ID不在属性文件中，随机选择另一个索引
        if self.attributes is not None and self.anno[index]['scene_id'] not in self.attributes:
            print(f"{self.anno[index]['scene_id']} not in attribute file!")
            return self.__getitem__(random.randint(0, len(self.anno)-1))

        # 获取或随机生成物体ID
        if "obj_id" in self.anno[index]:
            obj_id = int(self.anno[index]["obj_id"])
        else:
            obj_id = random.randint(0, self.max_obj_num - 1)
        # 获取问题和描述
        if 'prompt' not in self.anno[index]:
            question = random.choice(obj_caption_wid_prompt).replace('<id>', f"<OBJ{obj_id:03}>")
        else:
            question = self.anno[index]["prompt"]
        caption = self.anno[index]["caption"]
        # 获取场景信息
        (scene_id, scene_feat, scene_img_feat, scene_mask, scene_locs, assigned_ids, object_id, pred_id,
        coords, coords_float, feats, superpoints, object_sp_mask, semantic_label, instance_label,
        superpoint_centers, superpoint_features, instance_sp_masks,
        num_superpoints, batch_offsets, superpoint_semantic_labels) = self.get_anno(index)
        # 打乱数据中物体的ID顺序，增加训练的随机性
        caption = update_caption(caption, assigned_ids)
        question = update_caption(question, assigned_ids)
        # 本批次数据量
        return (scene_id, scene_feat, scene_img_feat, scene_mask, scene_locs, assigned_ids, obj_id, pred_id,
        caption, question, coords, coords_float, feats, superpoints, object_sp_mask, semantic_label,
        instance_label, superpoint_centers, superpoint_features, instance_sp_masks, num_superpoints, batch_offsets, superpoint_semantic_labels)


def train_collate_fn(batch):
    # 解包批次数据
    (scene_ids, scene_feats, scene_img_feats, scene_masks, scene_locs, assigned_ids, obj_ids, pred_ids, captions,
    questions, coordss, coords_floats, featss, superpointss, object_sp_masks, semantic_labels, instance_labels,
    superpoint_centerss, superpoint_featuress, instance_sp_maskss, num_superpointss, batch_offsets, superpoint_semantic_labels) = zip(*batch)
    # 使用pad_sequence处理变长序列：
    batch_scene_feat = pad_sequence(scene_feats, batch_first=True)
    batch_scene_img_feat = pad_sequence(scene_img_feats, batch_first=True)
    batch_scene_mask = pad_sequence(scene_masks, batch_first=True).to(torch.bool)
    batch_scene_locs = pad_sequence(scene_locs, batch_first=True)
    batch_assigned_ids = pad_sequence(assigned_ids, batch_first=True)
    # batch_detach_mask = torch.ones_like(batch_scene_mask, dtype=torch.bool)
    # for i in range(batch_detach_mask.shape[0]):
    #     batch_detach_mask[i][:detach_masks[i].shape[0]] = detach_masks[i]
    obj_ids = torch.tensor(obj_ids)
    pred_ids = torch.tensor(pred_ids)
    # batch_coords = pad_sequence(coordss, batch_first=True)
    # batch_coords_float = pad_sequence(coords_floats, batch_first=True)
    # batch_feats = pad_sequence(featss, batch_first=True)
    # batch_superpoints = pad_sequence(superpointss, batch_first=True)
    # batch_object_sp_masks = pad_sequence(object_sp_masks, batch_first=True)
    # batch_offsets = pad_sequence(batch_offsets, batch_first=True)
    return {
        "scene_id": scene_ids,
        "scene_feat": batch_scene_feat,
        "scene_img_feat": batch_scene_img_feat,
        "scene_mask": batch_scene_mask,
        "scene_locs": batch_scene_locs,
        "assigned_ids": batch_assigned_ids,
        # "detach_mask": batch_detach_mask,
        "obj_ids": obj_ids,
        "pred_ids": pred_ids,
        "answers": captions,
        "questions": questions,
        # "ref_captions": ref_captions,
        # "ids": index
        "coords": coordss,
        "coords_float": coords_floats,
        "feats": featss,
        "superpoints": superpointss,
        "object_sp_masks": object_sp_masks,
        "batch_offsets": batch_offsets,
        "superpoint_semantic_labels": superpoint_semantic_labels,
    }
