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

# pcd_dir = "/workspace/data/scannet/ChatScene/mask3d_ins_data_original/pcd_all"
pcd_dir = "/workspace/data/scannet/ChatScene/mask3d_ins_data/pcd_all"
scene_id = "scene0000_00"
pcd_path = os.path.join(pcd_dir, f"{scene_id}.pth")
data = torch.load(pcd_path)
(coords, colors, superpoint, sem_labels, instance_labels,mask3d_instance_class_labels,
mask3d_instance_segids,superpoint_centers, superpoint_features, instance_sp_masks,
instance_features, num_superpoints, coords_align) = data
object_id =2