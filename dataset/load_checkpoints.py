import torch

# 加载预训练权重
checkpoint = torch.load('checkpoints/spf_scannet_512.pth')

# 打印顶层键
print("Top level keys:")
print(checkpoint.keys())

# 如果有 'model' 键，查看模型权重的键
if 'model' in checkpoint:
    print("\nModel weights keys:")
    for key in checkpoint['model'].keys():
        print(key)

# 可以进一步查看具体的权重形状
print("\nWeights shapes:")
for key, value in checkpoint['model'].items():
    if isinstance(value, torch.Tensor):
        print(f"{key}: {value.shape}")