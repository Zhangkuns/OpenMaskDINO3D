import torch as th
from spconv.pytorch import SubMConv3d, SparseConvTensor

# 创建随机数据
xyz = th.randint(0, 100, (100, 4), dtype=th.int32, device='cuda')
xyz[:, 0] = 0  # 设置第一列为0
feat = th.rand(100, 32, device='cuda')

# 创建稀疏卷积张量
sp = SparseConvTensor(feat, xyz, (200, 200, 200), 1)
conv = SubMConv3d(32, 64, 3).cuda()
sp_features = conv(sp)  # 这里出错
print(sp_features)