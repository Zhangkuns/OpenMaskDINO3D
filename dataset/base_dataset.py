import logging
import os
import random

from torch.utils.data import Dataset
import torch
import re
import numpy as np
import math

from ChatQformer.models.point_extractor import PointExtractor

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):

    def __init__(self):

        self.media_type = "point_cloud"  # 数据类型为点云
        self.anno = None  # 标注数据
        self.attributes = None  # 属性数据
        self.feats = None  # 特征
        self.img_feats = None  # 图像特征
        self.scene_feats = None  # 场景特征
        self.scene_img_feats = None  # 场景图像特征
        self.scene_masks = None  # 场景掩码
        self.feat_dim = 1024  # 特征维度
        self.img_feat_dim = 1024  # 图像特征维度
        self.max_obj_num = 100  # 最大物体数量
        self.pcd_dir = ""
        self.aug = True  # 是否进行数据增强
        self.point_extractor = PointExtractor(
            input_channel=6,
            blocks=5,
            block_reps=2,
            media=32,
            normalize_before=True,
            return_blocks=True,
            pool='mean',
            fix_module= ['input_conv', 'unet', 'output_layer'],
            pretrained='checkpoints/spf_scannet_512.pth'  # 如果有预训练权重
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    # 场景特征准备方法
    def prepare_scene_features(self):
        # 获取所有场景ID
        if self.feats is not None:
            scan_ids = set('_'.join(x.split('_')[:2]) for x in self.feats.keys())
        else:
            scan_ids = set('_'.join(x.split('_')[:2]) for x in self.img_feats.keys())
        scene_feats = {}
        scene_img_feats = {}
        scene_masks = {}
        # 定义不需要的物体标签
        unwanted_words = ["wall", "ceiling", "floor", "object", "item"]
        for scan_id in scan_ids:
            if scan_id not in self.attributes:
                continue
            scene_attr = self.attributes[scan_id]
            # obj_num = scene_attr['locs'].shape[0]
            # 获取物体ID和标签
            obj_num = self.max_obj_num
            if 'obj_ids' in scene_attr:
                obj_ids = scene_attr['obj_ids']
            else :
                obj_ids = [_ for _ in range(obj_num)]
            if 'objects' in scene_attr:
                obj_labels = scene_attr['objects']
            else:
                obj_labels = [''] * obj_num
            # 初始化特征列表
            scene_feat = []
            scene_img_feat = []
            scene_mask = []
            # 处理每个物体
            for _i, _id in enumerate(obj_ids):
                item_id = '_'.join([scan_id, f'{_id:02}'])
                # 处理3D特征
                if self.feats is None or item_id not in self.feats:
                    # scene_feat.append(torch.randn((self.feat_dim)))
                    scene_feat.append(torch.zeros(self.feat_dim))
                else:
                    scene_feat.append(self.feats[item_id])
                # 处理图像特征
                if self.img_feats is None or item_id not in self.img_feats:
                    # scene_img_feat.append(torch.randn((self.img_feat_dim)))
                    scene_img_feat.append(torch.zeros(self.img_feat_dim))
                else:
                    scene_img_feat.append(self.img_feats[item_id].float())
                # if scene_feat[-1] is None or any(x in obj_labels[_id] for x in unwanted_words):
                #     scene_mask.append(0)
                # else:
                scene_mask.append(1)
            # 将列表转换为张量
            scene_feat = torch.stack(scene_feat, dim=0)
            scene_img_feat = torch.stack(scene_img_feat, dim=0)
            scene_mask = torch.tensor(scene_mask, dtype=torch.int)
            # 保存到字典
            scene_feats[scan_id] = scene_feat
            scene_img_feats[scan_id] = scene_img_feat
            scene_masks[scan_id] = scene_mask
        return scene_feats, scene_img_feats, scene_masks

    def get_anno(self, index):
        # 获取指定索引的场景ID
        scene_id = self.anno[index]["scene_id"]
        if self.attributes is not None:
            scene_attr = self.attributes[scene_id]
            # obj_num = scene_attr["locs"].shape[0]
            scene_locs = scene_attr["locs"]
        else:
            scene_locs = torch.randn((1, 6))
        # 获取对象ID
        object_id = int(self.anno[index]["obj_id"])
        # 获取Mask3D的对象ID
        pred_id = int(self.anno[index]["pred_id"])
        # 获取点云数据路径并加载
        pcd_path = os.path.join(self.pcd_dir, f"{scene_id}.pth")
        # 加载保存的数据
        # (coords, colors, superpoint, sem_labels, instance_labels,mask3d_instance_class_labels, mask3d_instance_segids,
        # superpoint_centers, superpoint_features, instance_sp_masks,instance_features, num_superpoints, coords_align) = torch.load(pcd_path)
        (coords, coords_align, colors, sem_labels, instance_labels, superpoint,
         superpoint_centers, superpoint_features, num_superpoints, superpoint_semantic_labels_nyu40,
         superpoint_semantic_labels_scannet20,
         mask3d_instance_class_labels, mask3d_instance_segids, mask3d_instance_sp_masks, mask3d_instance_features,
         gt_instance_labels, gt_segids
         ) = torch.load(pcd_path)
        object_sp_mask = mask3d_instance_sp_masks[pred_id] # 使用mask3d的实例分割结果
        # object_sp_mask = gt_instance_sp_masks[object_id] # 使用GT的实例分割结果
        # 准备提取点云特征
        xyz, xyz_middle, rgb, superpoint, semantic_label, instance_label = self.transform_train(coords, colors, superpoint,sem_labels, instance_labels)
        coord = torch.from_numpy(xyz).long()
        coord_float = torch.from_numpy(xyz_middle).float()
        feat = torch.from_numpy(rgb).float()
        superpoint = torch.from_numpy(superpoint)
        semantic_label = torch.from_numpy(semantic_label.astype(np.int64)).long()
        instance_label = torch.from_numpy(instance_label.astype(np.int64)).long()

        ann_ids, scan_ids, coords, coords_float, feats, superpoints, object_ids= [], [], [], [], [], [], []
        batch_offsets = [0]
        superpoint_bias = 0

        superpoint = superpoint + superpoint_bias
        superpoint_bias = superpoint.max().item() + 1
        batch_offsets.append(superpoint_bias)
        coords.append(torch.cat([torch.LongTensor(coord.shape[0], 1).fill_(0), coord], 1))
        coords_float.append(coord_float)
        feats.append(feat)
        superpoints.append(superpoint)

        batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)  # int [B+1]
        coords = torch.cat(coords, 0)  # long [B*N, 1 + 3], the batch item idx is put in b_xyz[:, 0]
        coords_float = torch.cat(coords_float, 0)  # float [B*N, 3]
        feats = torch.cat(feats, 0)  # float [B*N, 3]
        superpoints = torch.cat(superpoints, 0).long()  # long [B*N, ]
        feats = torch.cat((feats, coords_float), dim=1)
        spatial_shape = np.clip((coords.max(0)[0][1:] + 1).numpy(), 128, None)  # long [3]
        batch_size = 1

        # 获取场景特征
        scene_feat = self.scene_feats[scene_id]
        if scene_feat.ndim == 1:
            scene_feat = scene_feat.unsqueeze(0)

        # 获取图像特征和掩码
        scene_img_feat = self.scene_img_feats[scene_id] if self.scene_img_feats is not None else torch.zeros((scene_feat.shape[0], self.img_feat_dim))
        scene_mask = self.scene_masks[scene_id] if self.scene_masks is not None else torch.ones(scene_feat.shape[0], dtype=torch.int)
        # 生成随机排列的ID
        assigned_ids = torch.randperm(self.max_obj_num) # !!!
        return (scene_id, scene_feat, scene_img_feat, scene_mask, scene_locs, assigned_ids, object_id,
        pred_id, coords, coords_float, feats, superpoints, object_sp_mask, semantic_label,
        instance_label, superpoint_centers, superpoint_features, mask3d_instance_sp_masks, num_superpoints, batch_offsets, superpoint_semantic_labels_scannet20)

    def transform_train(self, xyz, rgb, superpoint, semantic_label, instance_label):
        # xyz_middle = xyz.copy()
        xyz_middle = self.data_aug(xyz, True, True, True)
        # 给RGB值添加随机噪声
        rgb += np.random.randn(3) * 0.1
        # 将点云坐标缩放50倍
        xyz = xyz_middle * 50
        # 将点云坐标归一化到非负范围
        xyz = xyz - xyz.min(0)
        return xyz, xyz_middle, rgb, superpoint, semantic_label, instance_label

    def data_aug(self, xyz, jitter=False, flip=False, rot=False):
        # jitter: 添加随机抖动
        # flip: X轴随机翻转
        # rot: 随机旋转
        m = np.eye(3)
        if jitter:
            m += np.random.randn(3, 3) * 0.1
        if flip:
            m[0][0] *= np.random.randint(0, 2) * 2 - 1  # flip x randomly
        if rot:
            theta = np.random.rand() * 2 * math.pi
            m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0],
                              [0, 0, 1]])  # rotation
        return np.matmul(xyz, m)
    

def update_caption(caption, assigned_ids):
    # 创建新旧ID的映射字典
    new_ids = {int(assigned_id): i for i, assigned_id in enumerate(assigned_ids)}
    # 匹配格式为<OBJxxx>的对象ID
    id_format = "<OBJ\\d{3}>"
    # 遍历所有匹配项，更新ID
    for match in re.finditer(id_format, caption):
        idx = match.start()
        old_id = int(caption[idx+4:idx+7])
        new_id = int(new_ids[old_id])
        caption = caption[:idx+4] + f"{new_id:03}" + caption[idx+7:]
    return caption

def recover_caption(caption, assigned_ids):
    # 输入参数：
    # caption: 包含对象ID标记的文本
    # assigned_ids: ID映射列表
    id_format = "<OBJ\\d{3}>"
    # 遍历匹配项：使用re.finditer查找所有匹配的ID标记 返回迭代器，包含所有匹配项的位置信息
    for match in re.finditer(id_format, caption):
        idx = match.start()
        new_id = int(caption[idx+4:idx+7])
        try:
            old_id = int(assigned_ids[new_id])
        except:
            old_id = random.randint(0, len(assigned_ids)-1)
        caption = caption[:idx+4] + f"{old_id:03}" + caption[idx+7:]
    return caption


if __name__ == "__main__":
    caption = "<OBJ001> <OBJ002>"
    assigned_ids = torch.randperm(5)
    print(assigned_ids)
    caption = update_caption(caption, assigned_ids)
    print(caption)
    caption = recover_caption(caption, assigned_ids)
    print(caption)
