import logging
import os
import random

from open3d.examples.geometry.voxel_grid_carving import xyz_spherical
from torch.utils.data import Dataset
import torch
import glob
from torch.nn.utils.rnn import pad_sequence
import re
import numpy as np
import math
import pointgroup_ops
import gc
from ChatQformer.dataset.point_extractor import PointExtractor
from ChatQformer.models.MaskDecoder import MaskDecoder
from ChatQformer.models.seg_loss import Criterion


def transform_train(xyz, rgb, superpoint, semantic_label, instance_label):

    # xyz_middle = xyz.copy()
    xyz_middle = data_aug(xyz,True, True, True)
    # 给RGB值添加随机噪声
    rgb += np.random.randn(3) * 0.1
    # 将点云坐标缩放50倍
    xyz = xyz_middle * 50
    # 将点云坐标归一化到非负范围
    xyz = xyz - xyz.min(0)
    return xyz, xyz_middle, rgb, superpoint, semantic_label, instance_label

def get_cropped_inst_label(instance_label: np.ndarray, valid_idxs: np.ndarray) -> np.ndarray:
        # 用于处理裁剪后的实例标签，确保标签连续且紧凑
        r"""
        get the instance labels after crop operation and recompact

        Args:
            instance_label (np.ndarray, [N]): instance label ids of point cloud
            valid_idxs (np.ndarray, [N]): boolean valid indices

        Returns:
            np.ndarray: processed instance labels
        """
        instance_label = instance_label[valid_idxs]
        j = 0
        while j < instance_label.max():
            if len(np.where(instance_label == j)[0]) == 0:
                instance_label[instance_label == instance_label.max()] = j
            j += 1
        return instance_label

def data_aug(xyz, jitter=False, flip=False, rot=False):
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
        m = np.matmul(m,[[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])  # rotation
    return np.matmul(xyz, m)

gc.collect()
torch.cuda.empty_cache()
pcd_dir = "/home/zhangkunshen/data/scannet/ChatScene/mask3d_ins_data/pcd_all"
scene_id = "scene0130_00"
pcd_path = os.path.join(pcd_dir, f"{scene_id}.pth")
coords, colors, superpoint, sem_labels, instance_labels,mask3d_instance_class_labels, mask3d_instance_segids,superpoint_centers, superpoint_features, instance_sp_masks,instance_features, num_superpoints,coords_align = torch.load(pcd_path)
object_id =2
object_sp_mask = instance_sp_masks[object_id]
# 使用掩码获取对应的特征
object_features = superpoint_features[object_id]
# 准备提取点云特征
xyz, xyz_middle, rgb, superpoint, semantic_label, instance_label = transform_train(coords, colors, superpoint, sem_labels, instance_labels)

coord = torch.from_numpy(xyz).long()
coord_float = torch.from_numpy(xyz_middle).float()
feat = torch.from_numpy(rgb).float()
superpoint = torch.from_numpy(superpoint)
semantic_label = torch.from_numpy(semantic_label.astype(np.int64)).long()
instance_label = torch.from_numpy(instance_label.astype(np.int64)).long()

ann_ids, scan_ids, coords, coords_float, feats, superpoints, object_ids, gt_pmasks, gt_spmasks, sp_ref_masks, lang_tokenss, lang_masks, lang_words, answerss, text_input_list = [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
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

voxel_coords, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(coords,batch_size , 4)

point_extractor = PointExtractor(
    input_channel=6,
    blocks=5,
    block_reps=2,
    media=32,
    normalize_before=True,
    return_blocks=True,
    pool='mean',
    fix_module= ['input_conv', 'unet', 'output_layer'],
    pretrained='checkpoints/spf_scannet_512.pth'  # 如果有预训练权重
).cuda()
batch = {
    'voxel_coords': voxel_coords.cuda(),  # 已经在 CUDA 上
    'p2v_map': p2v_map.cuda(),           # 需要移动到 CUDA
    'v2p_map': v2p_map.cuda(),           # 需要移动到 CUDA
    'spatial_shape': spatial_shape,       # 如果是张量需要移动到 CUDA
    'feats': feats.cuda(),               # 已经在 CUDA 上
    'superpoints': superpoints.cuda(),    # 需要移动到 CUDA
    'batch_offsets': batch_offsets.cuda() # 需要移动到 CUDA
}
point_features = point_extractor(batch)
point_features = point_features.unsqueeze(0).cuda()
text_features = torch.randn(batch_size, 1, 512).cuda()
batch_offsets = batch_offsets.cuda()
mask_decoder = MaskDecoder(media=32,            # 中间维度
        num_layer=6,         # 解码器层数
        d_model=256,         # 模型维度
        d_text=512,          # 文本特征维度
        nhead=8,             # 注意力头数
        hidden_dim=1024,     # 隐藏层维度
        dropout=0.0,         # dropout率
        activation_fn='gelu', # 激活函数
        attn_mask=True       # 是否使用注意力掩码
).cuda()
out = mask_decoder(point_features, batch_offsets, text_features)
criterion = Criterion(
     loss_weight = [1.0, 1.0, 0.5, 5.0],
     loss_fun='focal'
)
object_sp_mask = object_sp_mask.cuda()
seg_loss, log_vars = criterion(out, object_sp_mask)

# 打印各个参数的信息
print("\nParameter Information:")
print(f"voxel_coords shape: {voxel_coords.shape}, dtype: {voxel_coords.dtype}")
print(f"p2v_map shape: {p2v_map.shape}, dtype: {p2v_map.dtype}")
print(f"v2p_map shape: {v2p_map.shape}, dtype: {v2p_map.dtype}")
# print(f"Number of unique voxels: {len(torch.unique(voxel_coords, dim=0))}")
print(f"Max voxel coordinate: {voxel_coords.max(dim=0)[0]}")
print(f"Min voxel coordinate: {voxel_coords.min(dim=0)[0]}")

