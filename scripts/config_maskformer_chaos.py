# ========================= data ==========================

anno_root = "annotations"  # annotation dir
pc_encoder = "uni3d"
segmentor = "mask3d"
version = ""

gt_feat_file = f"{anno_root}/scannet_gt_{pc_encoder}_feats.pt"
seg_feat_file = f"{anno_root}/scannet_{segmentor}_{pc_encoder}_feats.pt"
seg_all_feat_file = f"{anno_root}/scannet_{segmentor}_{pc_encoder}_feats_all.pt"
gt_img_feat_file = f"{anno_root}/scannet_gt_videofeats.pt"
seg_img_feat_file = f"{anno_root}/scannet_{segmentor}_videofeats.pt"
seg_all_img_feat_file = f"{anno_root}/scannet_{segmentor}_videofeats_all.pt"
gt_train_attr_file = f"{anno_root}/scannet_train_attributes.pt"
gt_val_attr_file = f"{anno_root}/scannet_val_attributes.pt"
seg_train_attr_file = f"{anno_root}/scannet_{segmentor}_train_attributes.pt"
seg_val_attr_file = f"{anno_root}/scannet_{segmentor}_val_attributes.pt"
seg_all_attr_file = f"{anno_root}/scannet_{segmentor}_all_attributes.pt"

train_tag = "scanrefer#obj_align#scan2cap#scanqa#sqa3d"
val_tag = "scanrefer#scanqa#scan2cap#sqa3d"

train_file_dict = {
    'scanrefer': [
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f"{anno_root}/scanrefer_{segmentor}_train{version}.json"
    ],
    'scan2cap': [
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f"{anno_root}/scan2cap_{segmentor}_train{version}.json"
    ],
    'obj_align': [
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f"{anno_root}/obj_align_{segmentor}_train{version}.json"
    ],
    'scanqa': [
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f"{anno_root}/scanqa_{segmentor}_train{version}.json"
    ],
    'sqa3d': [
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f"{anno_root}/sqa3d_{segmentor}_train{version}.json"
    ]
}

val_file_dict = {
    'scanqa': [
        seg_feat_file,
        seg_img_feat_file,
        seg_val_attr_file,
        f"{anno_root}/scanqa_{segmentor}_val{version}.json"
    ],
    'scanrefer': [
        seg_feat_file,
        seg_img_feat_file,
        seg_val_attr_file,
        f"{anno_root}/scanrefer_{segmentor}_val{version}.json"
    ],
    'scan2cap': [
        seg_feat_file,
        seg_img_feat_file,
        seg_val_attr_file,
        f"{anno_root}/scan2cap_{segmentor}_val{version}.json"
    ],
    'sqa3d': [
        seg_feat_file,
        seg_img_feat_file,
        seg_val_attr_file,
        f"{anno_root}/sqa3d_{segmentor}_val{version}.json"
    ]
}



num_workers = 12
batch_size = 8


# ========================= model ==========================
model = dict(
    low_resource=True,
    max_txt_len=64,
    end_sym="</s>",
    system_path="prompts/system.txt",
    instruction_path="prompts/instruction.txt",
    role=("USER", "ASSISTANT"),
    llama_model_path="llm/vicuna-7b-v1.5",
    input_dim=1024,
    img_input_dim=1024,
    attr_dim=512,
    scene_dim=256,
    pos_dim=128,
    encoder_num_layers=3,
    add_scene_token=True,
    add_img_token=True,
    use_lora=True,
    train_emb=True,
    train_img_proj=False,
    no_obj=False,
    max_obj_num=100,
    bidirection=False,
    add_pos_emb=False,
    feat_fusion=False,
    fuse_with_id=False,
    use_objid=True,
    use_location_token=False,
    mask_decoder=dict(
        media=32,            # 中间维度
        num_layer=6,         # 解码器层数
        d_model=256,         # 模型维度
        d_text=512,          # 文本特征维度
        nhead=8,             # 注意力头数
        hidden_dim=1024,     # 隐藏层维度
        dropout=0.0,         # dropout率
        activation_fn='gelu', # 激活函数
        attn_mask=True       # 是否使用注意力掩码
    ),
    point_extractor=dict(
        input_channel=6,
        blocks=5,
        block_reps=2,
        media=32,
        normalize_before=True,
        return_blocks=True,
        pool='mean',
        fix_module= ['input_conv', 'unet', 'output_layer'],
        pretrained='checkpoints/spf_scannet_512.pth'
    ),
    criterion=dict(
     loss_weight = [1.0, 1.0, 0.5, 5.0],
     loss_fun='focal'
    )
)

lora = dict(
    lora_target_modules=[
      "q_proj",
      "v_proj",
      "k_proj",
      "o_proj",
      "gate_proj",
      "up_proj",
      "down_proj"
    ],
    lora_r=64,
    lora_alpha=16,
    lora_dropout=0.05
)

optimizer = dict(
    opt="adamW",
    lr=5e-3,
    opt_betas=[0.9, 0.999],  # default
    weight_decay=0.02,
    scaler_enable=False,
    max_grad_norm=5,  # requires a positive float, use -1 to disable
    # use a different lr for some modules, e.g., larger lr for new modules
    different_lr=dict(
        enable=False,
        module_names=["model.embed_tokens"],
        lr=[5e-4],
        wd=[0.02]
    ),
)

scheduler = dict(sched="cosine", epochs=3, min_lr_multi=0.01, warmup_epochs=0.1)

evaluate = False

# ========================= wandb ==========================
wandb = dict(
    enable=True,
    # entity="barbin-alexina-916-california-state-university-northridge-org",  # username or team name to store the runs, see https://docs.wandb.ai/ref/python/init
    project="Scene-LLM",
    name="experiment-2025-2-20",
)
dist_url = "env://"
device = "cuda"

# ========================= others ==========================
output_dir = "outputs/tmp"  # output dir
resume = True  # if True, load optimizer and scheduler states as well
debug = False
log_freq = 20
# eval_freq = 500
seed = 42
gpu_num = 1
save_latest = True
do_save = True
auto_resume = True
pretrained_path = "/workspace/data/checkpoints/ckpt_01_3446.pth"
img_projector_path = ""
