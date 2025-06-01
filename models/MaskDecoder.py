import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model=256, nhead=8, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, source, query, attn_mask=None, pe=None):
        """
        source (B, N_p, d_model)
        batch_offsets Tensor (b, n_p)
        query Tensor (b, n_q, d_model)
        attn_masks Tensor (b, n_q, n_p)
        """
        B = query.shape[0]
        query = self.with_pos_embed(query, pe)
        k = v = source
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.nhead, 1, 1).view(B * self.nhead, query.shape[1],
                                                                                k.shape[1])
            output, output_weight = self.attn(query, k, v, attn_mask=attn_mask)  # (1, 100, d_model)
        else:
            output, output_weight = self.attn(query, k, v)
        self.dropout(output)
        output = output + query
        self.norm(output)

        return output, output_weight  # (b, n_q, d_model), (b, n_q, n_v)


class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model=256, nhead=8, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.nhead = nhead
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, x, x_mask=None, attn_mask=None, pe=None):
        """
        x Tensor (b, n_w, c)
        x_mask Tensor (b, n_w)
        """
        B = x.shape[0]
        q = k = self.with_pos_embed(x, pe)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.nhead, 1, 1).view(B * self.nhead, q.shape[1], k.shape[1])
            output, _ = self.attn(q, k, x, key_padding_mask=x_mask, attn_mask=attn_mask)  # (1, 100, d_model)
        else:
            output, _ = self.attn(q, k, x, key_padding_mask=x_mask)
        output = self.dropout(output) + x
        output = self.norm(output)
        return output


class FFN(nn.Module):

    def __init__(self, d_model, hidden_dim, dropout=0.0, activation_fn='relu'):
        super().__init__()
        if activation_fn == 'relu':
            self.net = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model),
                nn.Dropout(dropout),
            )
        elif activation_fn == 'gelu':
            self.net = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model),
                nn.Dropout(dropout),
            )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        output = self.net(x)
        output = output + x
        output = self.norm(output)
        return output

class MaskDecoder(nn.Module):
    def __init__(
            self,
            # decoder=None,
            d_text=512,
            num_layer=6,
            d_model=256,
            nhead=8,
            hidden_dim=1024,
            dropout=0.0,
            activation_fn='gelu',
            attn_mask=True,
            media=32
    ):
        super().__init__()
        self.num_layer = num_layer
        # 输入特征投影
        self.input_proj = nn.Sequential(nn.Linear(media, d_model), nn.LayerNorm(d_model), nn.ReLU())
        # 语言特征投影
        self.lang_proj = nn.Linear(d_text, d_model)
        # Transformer层组件
        self.sa_layers = nn.ModuleList([])  # 自注意力层
        self.sa_ffn_layers = nn.ModuleList([])  # 自注意力后的前馈网络
        self.cross_attn_layers = nn.ModuleList([])  # 交叉注意力层
        self.ca_ffn_layers = nn.ModuleList([])  # 交叉注意力后的前馈网络
        # 循环构建每一层
        for i in range(num_layer):
            self.sa_layers.append(SelfAttentionLayer(d_model, nhead, dropout))
            self.sa_ffn_layers.append(FFN(d_model, hidden_dim, dropout, activation_fn))
            self.cross_attn_layers.append(CrossAttentionLayer(d_model, nhead, dropout))
            self.ca_ffn_layers.append(FFN(d_model, hidden_dim, dropout, activation_fn))
        # 输出归一化
        self.out_norm = nn.LayerNorm(d_model)
        # 分数预测层
        self.out_score = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))
        # 掩码特征投影
        self.x_mask = nn.Sequential(nn.Linear(media, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.attn_mask = attn_mask
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_mask(self, query, mask_feats):
        # 使用爱因斯坦求和计算预测掩码 计算查询特征和掩码特征的相似度
        pred_masks = torch.einsum('bnd,bmd->bnm', query, mask_feats)
        if self.attn_mask:
            # 将预测掩码转换为二值掩码（小于0.5的为True）
            attn_masks = (pred_masks.sigmoid() < 0.5).bool()  # [B, 1, num_sp]
            attn_masks[torch.where(attn_masks.sum(-1) == attn_masks.shape[-1])] = False
            attn_masks = attn_masks.detach()
        else:
            attn_masks = None
        return pred_masks, attn_masks

    def _forward_head(self, queries, mask_feats):
        norm_query = self.out_norm(queries)
        pred_scores = self.out_score(norm_query)
        pred_masks, attn_masks = self.get_mask(norm_query, mask_feats)
        return pred_scores, pred_masks, attn_masks

    def forward(self, sp_feats, batch_offsets, text_features=None, **kwargs):

        x = sp_feats
        batch_lap_pos_enc = None
        inst_feats = self.input_proj(x)  # 输入特征投影
        mask_feats = self.x_mask(x)  # 生成掩码特征
        queries = self.lang_proj(text_features)  # 文本特征投影
        pred_masks = []
        pred_scores = []
        B = len(batch_offsets) - 1

        # 0-th prediction 初始预测
        pred_score, pred_mask, attn_mask = self._forward_head(queries, mask_feats)
        pred_scores.append(pred_score)
        pred_masks.append(pred_mask)

        # multi-round
        for i in range(self.num_layer):
            # 交叉注意力处理
            queries, _ = self.cross_attn_layers[i](inst_feats, queries, attn_mask)
            queries = self.ca_ffn_layers[i](queries)
            # 自注意力处理
            queries = self.sa_layers[i](queries, None)
            queries = self.sa_ffn_layers[i](queries)
            # 新的预测
            pred_score, pred_mask, attn_masks = self._forward_head(queries, mask_feats)
            pred_scores.append(pred_score)
            pred_masks.append(pred_mask)

        # aux_outputs：生成辅助输出，用于存储每一层的中间结果（不包括最后一层）。
        # 对于每层的分类预测、分数和掩码，构造一个字典，并将它们加入到 aux_outputs 列表中。
        aux_outputs =[
            {'masks': masks, 'scores': scores}
            for scores, masks in zip(pred_scores[:-1], pred_masks[:-1])]

        return {
            'masks': pred_masks[-1],
            'scores': pred_scores[-1],
            'aux_outputs': aux_outputs
        }

        # return dict(
        #     masks=pred_masks[-1],
        #     scores=pred_scores[-1],
        #     aux_outputs=aux_outputs)

# class ScanNetQueryDecoder(MaskDecoder):
#     def __init__(
#             self,
#             num_semantic_classes = 19,
#             num_semantic_linears = 1,
#             d_model = 256,
#             **kwargs
#     ):
#         super().__init__(d_model=d_model, **kwargs)
#         assert num_semantic_linears in [1, 2]
#         if num_semantic_linears == 2:
#             self.out_sem = nn.Sequential(
#                 nn.Linear(d_model, d_model),
#                 nn.ReLU(),
#                 nn.Linear(d_model, num_semantic_classes + 1))
#         else:
#             self.out_sem = nn.Linear(d_model, num_semantic_classes + 1)
#
#     def _forward_head(self, queries, mask_feats, last_flag):
#         cls_preds, sem_preds, pred_scores, pred_masks, attn_masks = [], [], [], [], []
#         norm_query = self.out_norm(queries)
#         cls_preds = self.out_cls(norm_query)
#         if last_flag:
#             sem_preds = self.out_sem(norm_query)
#         pred_scores = self.out_score(norm_query)
#         pred_masks, attn_masks = self.get_mask(norm_query, mask_feats)
#         return cls_preds, sem_preds, pred_scores, pred_masks, attn_masks
#
#     def forward(self, sp_feats, batch_offsets, text_features=None, **kwargs):
#
#         x = sp_feats
#         batch_lap_pos_enc = None
#         inst_feats = self.input_proj(x)  # 输入特征投影
#         mask_feats = self.x_mask(x)  # 生成掩码特征
#         queries = self.lang_proj(text_features)  # 文本特征投影
#         cls_preds, sem_preds, pred_scores, pred_masks = [], [], [], []
#         B = len(batch_offsets) - 1
#
#         # 0-th prediction 初始预测
#         cls_pred, sem_pred, pred_score, pred_mask, attn_mask = self._forward_head(queries, mask_feats,last_flag=False)
#         cls_preds.append(cls_pred)
#         sem_preds.append(sem_pred)
#         pred_scores.append(pred_scores)
#         pred_masks.append(pred_masks)
#
#         # multi-round
#         for i in range(self.num_layer):
#             # 交叉注意力处理
#             queries, _ = self.cross_attn_layers[i](inst_feats, queries, attn_mask)
#             queries = self.ca_ffn_layers[i](queries)
#             # 自注意力处理
#             queries = self.sa_layers[i](queries, None)
#             queries = self.sa_ffn_layers[i](queries)
#             # 新的预测
#             last_flag = i == len(self.cross_attn_layers) - 1
#             cls_pred, sem_pred, pred_score, pred_mask, attn_masks = self._forward_head(queries, mask_feats, last_flag)
#             cls_preds.append(cls_pred)
#             sem_preds.append(sem_pred)
#             pred_scores.append(pred_score)
#             pred_masks.append(pred_mask)
#
#         # aux_outputs：生成辅助输出，用于存储每一层的中间结果（不包括最后一层）。
#         # 对于每层的分类预测、分数和掩码，构造一个字典，并将它们加入到 aux_outputs 列表中。
#         aux_outputs = [
#             dict(
#                 cls_preds=cls_pred,
#                 sem_preds=sem_pred,
#                 masks=masks,
#                 scores=scores)
#             for cls_pred, sem_pred, scores, masks in zip(
#                 cls_preds[:-1], sem_preds[:-1],
#                 pred_scores[:-1], pred_masks[:-1])]
#         return dict(
#             cls_preds=cls_preds[-1],
#             sem_preds=sem_preds[-1],
#             masks=pred_masks[-1],
#             scores=pred_scores[-1],
#             aux_outputs=aux_outputs)



