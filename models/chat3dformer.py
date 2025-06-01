import random
import logging
from abc import ABC

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import torch.nn.functional as F

from .modeling_llama import LlamaForCausalLM
from transformers import LlamaTokenizer, LlamaConfig
from ChatQformer.models.position_embedding import PositionEmbeddingCoordsSine
from peft import LoraConfig, get_peft_model
# from models.load_llama import init_llama_model
from torch.nn.utils.rnn import pad_sequence

import contextlib
from ChatQformer.dataset.base_dataset import update_caption, recover_caption

logger = logging.getLogger(__name__)


# nclamp 函数
def nclamp(input, min, max):
    return input.clamp(min=min, max=max).detach() + input - input.detach()

# 模型训练调试工具
# 打印每个参数的4项信息：
# 参数名
# 可训练状态(requires_grad)
# 梯度状态(是否有梯度)
# 参数形状
def print_grad_status(model):
    """Call this function after losses.backward()
    and it will find out all variables without grad, which
    means that the varaible is not in the graph.
    """
    for name, p in model.named_parameters():
        print('{:80s}{:20s}{:20s}{}'.format(name,
            '(Trainable)' if p.requires_grad else '(Fixed)',
            '(Has grad):' if p.grad is not None else '(No grad backward):',
            list(p.shape)))


class Chat3D(nn.Module):
    """
    VideoChat model.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        llama_model_path = config.model.llama_model_path
        self.low_resource = config.model.low_resource  # 低资源模式
        self.max_txt_len = config.model.max_txt_len    # 最大文本长度
        self.end_sym = config.model.end_sym  # 结束符
        self.system_path = config.model.system_path  # 系统文本路径
        self.instruction_path = config.model.instruction_path  # 指令文本路径
        self.role = config.model.role # 角色
        # 特征配置
        self.no_obj = config.model.no_obj
        self.add_scene_token = config.model.add_scene_token  # 添加场景token
        self.add_img_token = config.model.add_img_token  # 添加图像token
        self.train_emb = config.model.train_emb  # 是否训练嵌入层
        self.train_img_proj = config.model.train_img_proj # 是否训练图像投影层
        self.input_dim = config.model.input_dim # 3D点云的输入维度
        self.img_input_dim = config.model.img_input_dim # 图像输入维度
        self.attr_dim = config.model.attr_dim # 属性维度
        self.scene_dim = config.model.scene_dim # 场景维度
        self.pos_dim = config.model.pos_dim # transformer位置编码嵌入的维度
        self.max_obj_num = config.model.max_obj_num # 最大物体数量
        self.bidirection = config.model.bidirection # 双向注意力机制
        self.add_pos_emb = config.model.add_pos_emb # 添加transformer位置编码嵌入
        self.feat_fusion = config.model.feat_fusion # 特征融合, 即按照奇偶顺序嵌入特征
        self.fuse_with_id = config.model.fuse_with_id # 与ID融合, 增强理解
        self.use_location_token = config.model.use_location_token # 使用位置token


        self.debug = config.debug # 调试模式
        if not self.debug:
            logger.info('Loading LLaMA')
            self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model_path, use_fast=False, legacy=False)
            # self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

            if self.low_resource:
                self.llama_model = LlamaForCausalLM.from_pretrained(
                    llama_model_path,
                    torch_dtype=torch.bfloat16,
                    load_in_8bit=True,
                    device_map="auto",
                    attn_implementation="flash_attention_2"
                )
            else:
                self.llama_model = LlamaForCausalLM.from_pretrained(
                    llama_model_path,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2"
                )
            # print(torch.cuda.memory_allocated(device="cuda:0")/1e9)
            # self.llama_model = self.llama_model.to("cuda")
            # print(torch.cuda.memory_allocated(device="cuda:0")/1e9)
            # breakpoint()
            # 冻结LLaMA模型参数
            logger.info("freeze LLAMA")
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False

            if config.model.use_lora:
                def find_linear_layers(model, lora_target_modules):
                    cls = torch.nn.Linear # 指定要查找的是PyTorch的线性层
                    lora_module_names = set() # 用于存储符合条件的层的名称
                    for name, module in model.named_modules():
                        if (
                            isinstance(module, cls)
                            and all(
                                [
                                    x not in name
                                    for x in [
                                        "instance2embed",
                                        "hidden_state2query"
                                    ]
                                ]
                            )
                            and any([x in name for x in lora_target_modules])
                        ):
                            lora_module_names.add(name)
                    return sorted(list(lora_module_names)) # 返回经过排序的层名称列表
            
                lora_target_modules = find_linear_layers(self.llama_model, config.lora.lora_target_modules)

                lora_config = LoraConfig(
                    r=config.lora.lora_r,
                    lora_alpha=config.lora.lora_alpha,
                    target_modules=lora_target_modules,
                    lora_dropout=config.lora.lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                self.llama_model = get_peft_model(self.llama_model, lora_config)
                self.llama_model.print_trainable_parameters()
                # 启用lm_head权重梯度
                self.llama_model.model.lm_head.weight.requires_grad = True
                # 将权重转换为float格式
                self.llama_model.model.lm_head.weight.data = self.llama_model.model.lm_head.weight.data.float()
                self.llama_model.print_trainable_parameters()
                # 启用词嵌入权重梯度
                self.llama_model.model.model.embed_tokens.weight.requires_grad = True
                # 将权重转换为float格式
                self.llama_model.model.model.embed_tokens.weight.data = self.llama_model.model.model.embed_tokens.weight.data.float()
                self.llama_model.print_trainable_parameters()
            else:
                self.llama_model.lm_head.weight.requires_grad = True
                self.llama_model.lm_head.weight.data = self.llama_model.lm_head.weight.data.float()
                self.llama_model.model.embed_tokens.weight.requires_grad = True
                self.llama_model.model.embed_tokens.weight.data = self.llama_model.model.embed_tokens.weight.data.float()
            # 启用梯度检查点
            self.llama_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
            # 创建对象标识符token, 用于标识不同的对象
            objid_tokens = []
            for i in range(self.max_obj_num):
                objid_tokens.append(f"<OBJ{i:03}>")
            # 记录原始词汇表大小, 这个索引将作为新添加的对象token的起始位置
            self.objid_start_idx = self.ori_vocab_size = len(self.llama_tokenizer)
            # 将创建的对象标识符token添加到tokenizer的词汇表中
            self.llama_tokenizer.add_tokens(objid_tokens, special_tokens=True)
            # 保存添加新token后的词汇表大小，这将是对象token的结束索引
            self.objid_end_idx = len(self.llama_tokenizer)
            # 调整模型的词嵌入大小
            self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))
            
            # if self.use_location_token:
            #     location_tokens = ["<LOCATION>", "</LOCATION>"]
            #     for i in range(1000):
            #         location_tokens.append(f"<LOC{i:03}>")
            #     self.llama_tokenizer.add_tokens(location_tokens, special_tokens=True)
            #     self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))

            self.llama_dim = self.llama_model.config.hidden_size
            logger.info('Loading LLAMA Done')
        else:
            self.llama_model = None
            self.llama_dim = 4096

        # 特征投影网络，用于处理对象特征
        self.object_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.llama_dim),
            nn.GELU(),
            nn.Linear(self.llama_dim, self.llama_dim)
        )
        # 处理图像特征的投影网络
        self.object_img_proj = nn.Sequential(
            nn.Linear(self.img_input_dim, self.llama_dim),
            nn.GELU(),
            nn.Linear(self.llama_dim, self.llama_dim)
        )
        # 如果不训练图像投影层，则冻结图像投影层参数
        if not self.train_img_proj:
            for p in self.object_img_proj.parameters():
                p.requires_grad = False
        self.pos_embedding = PositionEmbeddingCoordsSine(d_pos=self.pos_dim)
        self.pos_proj = nn.Sequential(
            nn.Linear(self.pos_dim, self.llama_dim)
        )
        # self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.scene_dim, nhead=8, dim_feedforward=2048, dropout=0.05, norm_first=True, batch_first=True)
        # self.relation_module = nn.TransformerEncoder(self.encoder_layer, num_layers=config.model.encoder_num_layers)
        # self.scene_init_proj = nn.Sequential(
        #     nn.Linear(self.input_dim, self.scene_dim)
        # )
        # self.scene_proj = nn.Sequential(
        #     nn.Linear(self.scene_dim, self.llama_dim),
        #     # nn.GELU(),
        #     # nn.Linear(self.llama_dim, self.llama_dim)
        # )
        
        # if not self.add_scene_token:
        #     for p in self.relation_module.parameters():
        #         p.requires_grad = False
        #     for p in self.scene_init_proj.parameters():
        #         p.requires_grad = False
        #     for p in self.scene_proj.parameters():
        #         p.requires_grad = False
                

        with open(self.system_path, "r") as f:
            self.system = "\n".join([x.strip() for x in f.readlines()])
        with open(self.instruction_path, "r") as f:
            self.instruction = "\n".join([x.strip() for x in f.readlines()])

        # 如果不是调试模式, 则调用prepare_fixed_embed()方法准备嵌入向量
        if not self.debug:
            self.p_0_embed, self.p_1_embed = self.prepare_fixed_embed()
        self.last_embed = None
        
        print_grad_status(self)

    def get_objid_embeds(self):
        objid_start_idx = self.objid_start_idx
        objid_end_idx = self.objid_end_idx
        if self.config.model.use_lora:
            objid_embeds = self.llama_model.model.model.embed_tokens.weight[objid_start_idx:objid_end_idx] # max_obj_num * 4096
        else:
            objid_embeds = self.llama_model.model.embed_tokens.weight[objid_start_idx:objid_end_idx]
        return objid_embeds
    
    def llama_embed_tokens(self, token_ids):
        if self.config.model.use_lora:
            return self.llama_model.model.model.embed_tokens(token_ids)
        else:
            return self.llama_model.model.embed_tokens(token_ids)

    def prepare_fixed_embed(self):
        # 这一行将系统提示(system)、指令(instruction)和角色(role)组合成一个完整的提示文本
        prompt = self.system + " " + self.instruction + " " + self.role[0] + ": " 
        p_0, p_1 = prompt.split("<REPLACE>")
        p_0_token = self.llama_tokenizer(p_0, return_tensors="pt", add_special_tokens=True)
        p_1_token = self.llama_tokenizer(p_1, return_tensors="pt", add_special_tokens=False)
        p_0_embed = self.llama_embed_tokens(p_0_token.input_ids).squeeze(0).detach()
        p_1_embed = self.llama_embed_tokens(p_1_token.input_ids).squeeze(0).detach()
        return p_0_embed, p_1_embed

    def get_text_emb(self, text, device="cpu"):
        # 使用LLaMA分词器将输入文本转换为张量，不添加特殊标记，并移至指定设备
        text_tokens = self.llama_tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
        # 通过LLaMA的词嵌入层将token ID转换为嵌入向量
        embeds = self.llama_embed_tokens(text_tokens.input_ids)
        if self.train_emb:
            # 标记新增词汇（ID大于原始词表大小）的位置，转换为0 / 1 张量并增加维度
            indices = text_tokens.input_ids >= self.ori_vocab_size
            indices = (indices * 1).unsqueeze(-1)
            # 原始词汇的嵌入冻结(detach)，新词汇的嵌入保持梯度
            embeds = (1 - indices) * embeds.detach() + indices * embeds
        else:
            embeds = embeds.detach()
        return embeds

    def encode_object_feat(self, feat, img_feat, locs):
        feat = torch.nn.functional.normalize(feat, dim=-1)
        img_feat = torch.nn.functional.normalize(img_feat, dim=-1)
        return feat, img_feat
    
    @staticmethod
    def get_dist_attention(pos, dist_exp=1):
        # pos (bs, obj_num, 3)
        dist = pos.unsqueeze(1) - pos.unsqueeze(2)
        dist = torch.sum(dist.abs()**dist_exp, dim=-1)
        dist_attn = torch.nn.functional.softmax(-dist, dim=-1)
        return dist_attn

    def get_object_list_embed(self, embed_obj, embed_img, embed_scene, scene_mask, obj_id, assigned_ids):
        valid_ids = torch.where(scene_mask)[0].tolist()
        if self.config.model.use_lora:
            objid_embeds = self.llama_model.model.model.embed_tokens.weight[self.objid_start_idx:self.objid_end_idx] # max_obj_num * 4096
        else:
            objid_embeds = self.llama_model.model.embed_tokens.weight[self.objid_start_idx:self.objid_end_idx]

        assigned_ids = assigned_ids[valid_ids]
        # 如果不训练则冻结嵌入
        if not self.train_emb:
            objid_embeds = objid_embeds.detach()
        selected_objid_embeds = objid_embeds[valid_ids]
        # 使用交错融合，偶数位置存对象特征，奇数位置存图像特征
        if self.use_location_token:
            # shape[0]：选中对象的数量（对象个数) shape[1]：每个嵌入向量的维度
            object_list_embed = torch.zeros((selected_objid_embeds.shape[0] * 2, selected_objid_embeds.shape[1]), dtype=selected_objid_embeds.dtype, device=selected_objid_embeds.device)
            object_list_embed[0::2, :] += embed_obj[assigned_ids]
            object_list_embed[1::2, :] += embed_img[assigned_ids]
            return object_list_embed
        # 加和融合，直接将ID嵌入、对象特征、位置信息相加
        if self.fuse_with_id:
            object_list_embed = selected_objid_embeds
            if not self.no_obj:
                object_list_embed += embed_obj[assigned_ids]
            if self.add_img_token:
                object_list_embed += embed_img[assigned_ids]
            return object_list_embed
        # 使用交错融合，偶数位置存ID特征嵌入，奇数位置存图像特征，对象特征
        if self.feat_fusion:
            object_list_embed = torch.zeros((selected_objid_embeds.shape[0] * 2, selected_objid_embeds.shape[1]), dtype=selected_objid_embeds.dtype, device=selected_objid_embeds.device)
            object_list_embed[0::2, :] = selected_objid_embeds
            if not self.no_obj:
                object_list_embed[1::2, :] += embed_obj[assigned_ids]
            if self.add_img_token:
                object_list_embed[1::2, :] += embed_img[assigned_ids]
            return object_list_embed
        # 没有对象特征，只有ID特征
        if self.no_obj:
            object_list_embed = torch.zeros((selected_objid_embeds.shape[0] * 2, selected_objid_embeds.shape[1]), dtype=selected_objid_embeds.dtype, device=selected_objid_embeds.device)
            object_list_embed[0::2, :] = selected_objid_embeds
            object_list_embed[1::2, :] = embed_img[assigned_ids]
            return object_list_embed
        # 图像特征和场景特征都没有
        if embed_img is None and embed_scene is None:
            object_list_embed = torch.zeros((selected_objid_embeds.shape[0] * 2, selected_objid_embeds.shape[1]), dtype=selected_objid_embeds.dtype, device=selected_objid_embeds.device)
            object_list_embed[0::2, :] = selected_objid_embeds
            object_list_embed[1::2, :] = embed_obj[assigned_ids]
            return object_list_embed
        # 图像特征没有，场景特征有
        if embed_img is None and embed_scene is not None:
            object_list_embed = torch.zeros((selected_objid_embeds.shape[0] * 3, selected_objid_embeds.shape[1]), dtype=selected_objid_embeds.dtype, device=selected_objid_embeds.device)
            object_list_embed[0::3, :] = selected_objid_embeds
            object_list_embed[1::3, :] = embed_obj[assigned_ids]
            object_list_embed[2::3, :] = embed_scene[assigned_ids]
            return object_list_embed
        # 图像特征有，场景特征没有
        if embed_img is not None and embed_scene is None:
            object_list_embed = torch.zeros((selected_objid_embeds.shape[0] * 3, selected_objid_embeds.shape[1]), dtype=selected_objid_embeds.dtype, device=selected_objid_embeds.device)
            object_list_embed[0::3, :] = selected_objid_embeds
            object_list_embed[1::3, :] = embed_obj[assigned_ids]
            object_list_embed[2::3, :] = embed_img[assigned_ids]
            return object_list_embed
        # 图像特征和场景特征都有
        if embed_img is not None and embed_scene is not None:
            object_list_embed = torch.zeros((selected_objid_embeds.shape[0] * 4, selected_objid_embeds.shape[1]), dtype=selected_objid_embeds.dtype, device=selected_objid_embeds.device)
            object_list_embed[0::4, :] = selected_objid_embeds
            object_list_embed[1::4, :] = embed_obj[assigned_ids]
            object_list_embed[2::4, :] = embed_scene[assigned_ids]
            object_list_embed[3::4, :] = embed_img[assigned_ids]
            return object_list_embed
        # 默认返回对象ID特征
        object_list_embed = selected_objid_embeds
        return object_list_embed

    def get_min_max_coord(self, xyz, scene_mask):
        # 获取3D场景中有效区域（由mask指定）的坐标最小值和最大值
        scene_mask = scene_mask.unsqueeze(-1).expand_as(xyz)
        masked_xyz_min = torch.where(scene_mask, xyz, torch.full_like(xyz, float('inf')))
        masked_xyz_max = torch.where(scene_mask, xyz, torch.full_like(xyz, float('-inf')))
        mins = masked_xyz_min.min(dim=1)[0]
        maxs = masked_xyz_max.max(dim=1)[0]
        return mins, maxs

    def forward_train(self, scene_feat, scene_img_feat, scene_locs, scene_mask, obj_ids, assigned_ids, questions, answers, is_eval=False, **kwargs):
        # 归一化对象特征和图像特征
        object_embed = torch.nn.functional.normalize(scene_feat, dim=-1)
        object_img_embed = torch.nn.functional.normalize(scene_img_feat, dim=-1)
        device = object_embed.device
        batch_size = object_embed.shape[0]
        proj_object_embed = self.object_proj(object_embed)
        proj_object_img_embed = self.object_img_proj(object_img_embed)
        # 添加位置编码
        if self.add_pos_emb:
            mins, maxs = self.get_min_max_coord(scene_locs[:, :, :3], scene_mask)
            pos_embed = self.pos_embedding(scene_locs[:, :, :3], input_range=[mins, maxs]) / 10
            proj_pos_embed = self.pos_proj(pos_embed)
            proj_object_embed = proj_object_embed + proj_pos_embed
            proj_object_img_embed = proj_object_img_embed + proj_pos_embed

        proj_scene_embed = None
        if self.add_scene_token:  # remember to change the evaluate 
            # if self.add_img_token:
            #     object_embed = object_embed + object_img_embed
            obj_embed = self.scene_init_proj(object_embed)
            mins, maxs = self.get_min_max_coord(scene_locs[:, :, :3], scene_mask)
            pos_embed = self.pos_embedding(scene_locs[:, :, :3], input_range=[mins, maxs])
            pos_embed = self.pos_proj(pos_embed)
            scene_embed = obj_embed + pos_embed
            scene_embed = self.relation_module(scene_embed, src_key_padding_mask=~scene_mask)
            proj_scene_embed = self.scene_proj(scene_embed)
        
        input_embed_list, attn_list, target_list = [], [], []
        max_seq_len = 0
        p_0_embed = self.p_0_embed.to(device)
        p_1_embed = self.p_1_embed.to(device)
        object_list_intervals = []

        # 遍历每个问题
        for i, question in enumerate(questions):
            prompt = f"{question} {self.role[1]}: "
            prompt_embed = self.get_text_emb(prompt, device = device.type).squeeze(0)
            # 获取对象列表嵌入
            object_list_embed = self.get_object_list_embed(
                proj_object_embed[i], 
                proj_object_img_embed[i] if self.add_img_token else None, 
                proj_scene_embed[i] if self.add_scene_token else None, 
                scene_mask[i],
                obj_ids[i],
                assigned_ids[i]
            )
            # object_list_embed = nclamp(object_list_embed, min=-0.05, max=0.05)
            # 记录对象列表在序列中的起始和结束位置
            object_list_intervals.append((p_0_embed.shape[0], p_0_embed.shape[0] + object_list_embed.shape[0]))
            # 将对象列表嵌入与其他嵌入拼接
            wrapped_embed = torch.cat([p_0_embed, object_list_embed, p_1_embed, prompt_embed], dim=0)
            # 创建注意力掩码
            wrapped_attn = torch.ones(wrapped_embed.size()[:-1], dtype=torch.long).to(wrapped_embed.device)
            empty_target = (torch.ones(wrapped_attn.shape[0], dtype=torch.long).to(device).fill_(-100))
            # 处理答案
            answer = answers[i] + self.end_sym
            to_regress_token = self.llama_tokenizer(answer, return_tensors="pt", add_special_tokens=False).to(device)
            # breakpoint()
            answer_target = to_regress_token.input_ids.masked_fill(to_regress_token.input_ids == self.llama_tokenizer.pad_token_id, -100).squeeze(0)
            # to_regress_embed = self.llama_model.model.embed_tokens(to_regress_token.input_ids).squeeze(0).detach()
            # 获取答案的嵌入
            to_regress_embed = self.get_text_emb(answer, device = device.type).squeeze(0)
            # 组合空目标和答案目标
            # 输入序列结构是：[前缀 + 对象列表 + 后缀 + 提示词 + 答案]
            # 我们只想让模型预测答案部分，而不是预测前面的内容
            # 通过将non-answer部分的target设为-100，模型在训练时就不会试图预测这些位置的token
            target = torch.cat([empty_target, answer_target], dim=0)
            input_embed = torch.cat([wrapped_embed, to_regress_embed], dim=0)
            attn = torch.cat([wrapped_attn, to_regress_token.attention_mask[0]], dim=0)
            # 保存到列表
            input_embed_list.append(input_embed)
            attn_list.append(attn)
            target_list.append(target)
            max_seq_len = max(max_seq_len, target.shape[0])
        
        max_seq_len = min(768, max_seq_len)

        def pad_and_trim(tensor_list, max_len, batch_first=True, padding_value=0):
            padded = pad_sequence(tensor_list, batch_first=batch_first, padding_value=padding_value)
            if padded.shape[1] > max_len:
                return padded[:, :max_len]
            return padded

        # 将输入序列、目标序列和注意力掩码裁剪填充到相同长度
        input_embeds = pad_and_trim(input_embed_list, max_seq_len, batch_first=True, padding_value=0).to(device)
        targets = pad_and_trim(target_list, max_seq_len, batch_first=True, padding_value=-100).to(device)
        attention_mask = pad_and_trim(attn_list, max_seq_len, batch_first=True, padding_value=0).to(device)
        # 如果启用双向注意力机制，则创建一个因果掩码
        if self.bidirection:
            input_dtype = input_embeds.dtype
            # 创建基础因果掩码（causal mask）
            causal_mask = torch.ones((max_seq_len, max_seq_len), dtype=input_dtype, device=device)
            causal_mask = torch.tril(causal_mask, diagonal=0)
            # 扩展掩码维度以匹配输入维度
            causal_mask = causal_mask[None, None, :, :].expand(input_embeds.shape[0], 1, -1, -1).clone()
            # 创建填充掩码
            padding_mask = causal_mask[..., :].eq(1.0) * attention_mask[:, None, None, :].eq(0.0)
            causal_mask[..., :] = causal_mask[..., :].masked_fill(padding_mask, 0.0)
            # 处理对象列表区间
            for i in range(causal_mask.shape[0]):
                st, ed = object_list_intervals[i]
                causal_mask[i, :, st:ed, st:ed] = 1.0
            attention_mask = causal_mask
        
        # label_weights = torch.ones(self.llama_model.config.vocab_size, device=device)
        # label_weights[self.objid_start_idx:self.objid_end_idx] = 10

        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
                # label_weights=label_weights
            )

        return dict(
            loss=outputs.loss, # 模型的训练损失
            obj_norm=proj_object_embed.norm(dim=-1).mean().detach().cpu(), # 物体嵌入的平均范数
            obj_img_norm=proj_object_img_embed.norm(dim=-1).mean().detach().cpu(), # 物体图像嵌入的平均范数
            objid_norm=self.get_objid_embeds().norm(dim=-1).mean().detach().cpu(), # 物体ID嵌入的平均范数
            scene_norm=proj_scene_embed.norm(dim=-1).mean().detach().cpu() if proj_scene_embed is not None else 0.,
            max_seq_len=max_seq_len # 最大序列长度
        )

    def evaluate(self, scene_feat, scene_img_feat, scene_locs, scene_mask, custom_prompt, obj_ids, assigned_ids, is_eval=True, **kwargs):
        # 特征编码
        object_embed, object_img_embed = self.encode_object_feat(scene_feat, scene_img_feat, scene_locs)
        device = object_embed.device
        batch_size, obj_num = object_embed.shape[:2]
        proj_object_embed = self.object_proj(object_embed)
        proj_object_img_embed = self.object_img_proj(object_img_embed)
        # 添加位置编码
        if self.add_pos_emb:
            mins, maxs = self.get_min_max_coord(scene_locs[:, :, :3], scene_mask)
            pos_embed = self.pos_embedding(scene_locs[:, :, :3], input_range=[mins, maxs]) / 10
            proj_pos_embed = self.pos_proj(pos_embed)
            proj_object_embed = proj_object_embed + proj_pos_embed
            proj_object_img_embed = proj_object_img_embed + proj_pos_embed
        # 场景token处理
        if self.add_scene_token:
            # if self.add_img_token:
            #     object_embed = object_embed + object_img_embed
            obj_embed = self.scene_init_proj(object_embed)
            mins, maxs = self.get_min_max_coord(scene_locs[:, :, :3], scene_mask)
            pos_embed = self.pos_embedding(scene_locs[:, :, :3], input_range=[mins, maxs])
            pos_embed = self.pos_proj(pos_embed)
            scene_embed = obj_embed + pos_embed
            scene_embed = self.relation_module(scene_embed, src_key_padding_mask=~scene_mask)
            proj_scene_embed = self.scene_proj(scene_embed)

        output_texts = []
        p_0_embed = self.p_0_embed.to(device).unsqueeze(0)
        p_1_embed = self.p_1_embed.to(device).unsqueeze(0)
        # 遍历每个问题
        for i in range(batch_size):
            # 构建提示词
            tmp_prompt = f" {custom_prompt[i]} {self.role[1]}: "
            tmp_prompt = update_caption(tmp_prompt, assigned_ids[i])
            prompt_embed = self.get_text_emb(tmp_prompt, device=device.type)
            # 获取对象列表嵌入
            object_list_embed = self.get_object_list_embed(
                proj_object_embed[i], 
                proj_object_img_embed[i] if self.add_img_token else None, 
                proj_scene_embed[i] if self.add_scene_token else None, 
                scene_mask[i],
                obj_ids[i],
                assigned_ids[i]
            )
            object_list_embed = object_list_embed.unsqueeze(0)
            # 组合所有嵌入
            wrapped_embed = torch.cat([p_0_embed, object_list_embed, p_1_embed, prompt_embed], dim=1)
            attention_mask=None
            if self.bidirection:
                seq_len = wrapped_embed.shape[1]
                attention_mask = torch.ones((seq_len, seq_len), dtype=wrapped_embed.dtype, device=device)
                attention_mask = torch.tril(attention_mask, diagonal=0)
                attention_mask = attention_mask[None, None, :, :].expand(1, 1, -1, -1).clone()
                st, ed = p_0_embed.shape[1], p_0_embed.shape[1] + object_list_embed.shape[1]
                attention_mask[:, :, st:ed, st:ed] = 1.0
            
            with self.maybe_autocast():
                # 文本生成
                outputs = self.llama_model.generate(
                    inputs_embeds=wrapped_embed,
                    max_new_tokens=self.max_txt_len,
                    # stopping_criteria=stopping_criteria,
                    num_beams=5,
                    # do_sample=True,
                    min_length=1,
                    # top_p=0.9,
                    repetition_penalty=3.0,
                    length_penalty=1,
                    temperature=1.0,
                    customized_mask=attention_mask
                )
            output_token = outputs[0]
            # 解码生成的token
            output_text = self.llama_tokenizer.decode(output_token)
            # 清理和格式化文本
            output_text = output_text.split(self.end_sym)[0]
            output_text = output_text.replace('  ', ' ').replace(' .', '.').strip()
            # 恢复caption中的ID信息
            output_text = recover_caption(output_text, assigned_ids[i].tolist())
            output_texts.append(output_text)
        return output_texts

    def forward(self, **kwargs):
        if "answers" in kwargs:
            return self.forward_train(**kwargs)
        if "custom_prompt" in kwargs:
            return self.evaluate(**kwargs)
        return None

    def _get_text_len(self, text):
        return self.llama_tokenizer(text, return_tensors="pt").input_ids.shape[1]

    def maybe_autocast(self, dtype=torch.bfloat16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")
        # 在GPU上：返回启用了AMP的上下文管理器
        # 在CPU上：返回空上下文管理器（不做任何操作）
        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    @property
    def device(self):
        return list(self.parameters())[0].device
