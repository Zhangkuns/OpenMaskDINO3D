import logging
import pointgroup_ops
import torch
import torch.nn as nn
import numpy as np
from transformers import LlamaTokenizer

outputs = torch.load('/workspace/work/Mask3Dformer/ChatQformer/debug/debug_outputs.pt')
hidden_states = outputs.hidden_states
# 1. 收集所有最后一层的隐藏状态
last_layer_states = []
for step_states in outputs['hidden_states'][1:]:
    last_layer_states.append(step_states[-1])
# 2. 拼接成一个完整的 tensor
all_hidden_states = torch.cat(last_layer_states, dim=1)  # 应该得到 (1,seq_len-1,4096)
# 3. 求均值
output_token = outputs['sequences']
seg_mask = outputs["sequences"][:,1:] == 2999 # 形状是 (1, seq_len)
seg_out = all_hidden_states

if not seg_mask.any():
    print("No SEG token found in generation")
    # 使用序列第一个token的特征作为默认值
    seg_features = seg_out[:, 0, :]
else:
    # 使用所有SEG token位置的特征
    seg_out_valid = seg_out[seg_mask]
    seg_features = seg_out_valid.mean(dim=0, keepdim=True)
llama_tokenizer = LlamaTokenizer.from_pretrained("llm/vicuna-7b-v1.5", use_fast=False, legacy=False)
# 解码生成的token
output_text = llama_tokenizer.decode(output_token)
# 清理和格式化文本
output_text = output_text.split(self.end_sym)[0]
output_text = output_text.replace('  ', ' ').replace(' .', '.').strip()
# 恢复caption中的ID信息
output_text = recover_caption(output_text, assigned_ids[i].tolist())
output_texts.append(output_text)