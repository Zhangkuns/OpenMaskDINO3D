import functools
import pointgroup_ops
import spconv.pytorch
import torch
import torch.nn as nn
#import spconv
from torch_scatter import scatter_max, scatter_mean
from ChatQformer.models.backbone import ResidualBlock, UBlock


class PointExtractor(nn.Module):
    def __init__(
            self,
            input_channel: int = 6,  # 输入通道数（默认6，可能是xyz + rgb）
            blocks: int = 5,  # U-Net块数
            block_reps: int = 2,  # 每个块中的重复次数
            media: int = 32,  # 中间特征维度
            normalize_before=True,  # 是否在前面进行归一化
            return_blocks=True,  # 是否返回所有块的特征
            pool='mean',  # 池化方式（平均或最大）
            fix_module=None,  # 需要冻结的模块
            pretrained=None  # 预训练模型路径
    ):
        super().__init__()

        # backbone and pooling
        if fix_module is None:
            fix_module = ['input_conv', 'unet', 'output_layer']
        self.input_conv = spconv.pytorch.SparseSequential(
            spconv.pytorch.SubMConv3d(
                input_channel,
                media,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key='subm1',
            )).cuda()
        block = ResidualBlock
        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)
        block_list = [media * (i + 1) for i in range(blocks)]
        self.unet = UBlock(
            block_list,
            norm_fn,
            block_reps,
            block,
            indice_key_id=1,
            normalize_before=normalize_before,
            return_blocks=return_blocks,
        ).cuda()
        self.output_layer = spconv.pytorch.SparseSequential(
            norm_fn(media),
            nn.ReLU(inplace=True)
        ).cuda()
        self.pool = pool

        self.init_weights()

        # 加载预训练权重
        if pretrained is not None:
            checkpoint = torch.load(pretrained)
            state_dict = checkpoint['model']  # 直接使用 model 部分的权重
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            print(f'Loaded pretrained model from {pretrained}')
            print(f"Missing keys: {len(missing_keys)}")  # 模型需要但权重文件中没有的键
            print(f"Unexpected keys: {len(unexpected_keys)}")  # 权重文件中有但模型不需要的键

            # # 冻结所有参数
            self.eval()  # 设置为评估模式
            for param in self.parameters():
                param.requires_grad = False

        for module in fix_module:
            if '.' in module:
                module, params = module.split('.')
                module = getattr(self, module)
                params = getattr(module, params)
                for param in params.parameters():
                    param.requires_grad = False
            else:
                module = getattr(self, module)
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, batch):
        return self.feat(**batch)

    def feat(self, voxel_coords, p2v_map, v2p_map, spatial_shape, feats, superpoints, batch_offsets, **kwargs):
        batch_size = len(batch_offsets) - 1
        # 体素化：将点云特征转换为体素特征
        voxel_feats = pointgroup_ops.voxelization(feats, v2p_map)
        # 创建稀疏张量
        x = spconv.pytorch.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)
        # 通过网络各层处理
        x = self.input_conv(x)
        x, _ = self.unet(x)
        x = self.output_layer(x)
        # 将特征映射回点云
        x = x.features[p2v_map.long()]  # (B*N, media)
        # superpoint pooling
        # 超点池化
        if self.pool == 'mean':
            x = scatter_mean(x, superpoints, dim=0)  # (B*M, media)
        elif self.pool == 'max':
            x, _ = scatter_max(x, superpoints, dim=0)  # (B*M, media)
        return x
