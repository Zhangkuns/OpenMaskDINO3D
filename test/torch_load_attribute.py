import numpy as np
import json
import sys

import torch
import random
from tqdm import tqdm
from collections import defaultdict
import argparse
from ChatQformer.utils.box_utils import get_box3d_min_max, box3d_iou, construct_bbox_corners
from ChatQformer.prompts.prompts import grounding_prompt
import string


segmentor="mask3d"
split="train"
instance_attribute_file = f"annotations/scannet_{segmentor}_{split}_attributes.pt"
scannet_attribute_file = f"annotations/scannet_{split}_attributes.pt"
instance_attrs = torch.load(instance_attribute_file, map_location='cpu')
scannet_attrs = torch.load(scannet_attribute_file, map_location='cpu')
object_id = 2
