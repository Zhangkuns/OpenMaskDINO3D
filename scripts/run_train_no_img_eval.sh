# 设置工作目录
cd /home/zhangkunshen/Mask3Dformer/ChatQformer
# 添加项目根目录到 PYTHONPATH
export PYTHONPATH=/home/zhangkunshen/Mask3Dformer:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=2
epoch=3
batch_size=3
lr=5e-6
train_emb=True
train_img_proj=True
add_img_token=False
add_scene_token=False
no_obj=False
input_dim=1024 # 1024
bidirection=False
different_lr=False
max_obj_num=100
lora_r=16
lora_alpha=16
add_pos_emb=False
feat_fusion=False
fuse_with_id=False
config=""
max_grad_norm=0.01
seed=16
use_location_token=False
segmentor="mask3d"
pc_encoder="uni3d"
llama_model_path="llm/vicuna-7b-v1.5"

train_tag="scanrefer#obj_align#scan2cap#scanqa#sqa3d"
#train_tag="scanrefer"
val_tag="scanrefer#scanqa#scan2cap#sqa3d"
#val_tag="scanqa#scan2cap#sqa3d#multi3dref"

evaluate=True
debug=True
if [ $debug = "True" ]; then
    enable_wandb=True
    gpu_num=1
    do_save=True
    other_info="debug"
else
    enable_wandb=False
    gpu_num=4
    do_save=True
    other_info="chatscene"
fi

tag="${train_tag}__${val_tag}__${other_info}"

pretrained_path="/home/zhangkunshen/Mask3Dformer/ChatQformer/outputs/20250221_013455_lr5e-6_ep3_scanrefer#obj_align#scan2cap#scanqa#sqa3d__scanqa#scan2cap#sqa3d__debug/ckpt_02_49646.pth"

OUTPUT_DIR=outputs/"$(date +"%Y%m%d_%H%M%S")"_lr"$lr"_ep"$epoch"_"$tag"
mkdir -p ${OUTPUT_DIR}

python tasks/train.py \
    "./scripts/config_maskformer.py" \
    output_dir "$OUTPUT_DIR" \
    scheduler.epochs "$epoch" \
    optimizer.lr "$lr" \
    model.add_scene_token "$add_scene_token" \
    model.add_img_token "$add_img_token" \
    pretrained_path "$pretrained_path" \
    evaluate "$evaluate" \
    wandb.enable "$enable_wandb" \
    gpu_num "$gpu_num" \
    do_save "$do_save" \
    batch_size "$batch_size" \
    model.train_emb "$train_emb" \
    model.train_img_proj "$train_img_proj" \
    train_tag "$train_tag" \
    val_tag "$val_tag" \
    model.no_obj "$no_obj" \
    segmentor "$segmentor" \
    pc_encoder "$pc_encoder" \
    model.input_dim "$input_dim" \
    model.bidirection "$bidirection" \
    optimizer.different_lr.enable "$different_lr" \
    model.max_obj_num "$max_obj_num" \
    lora.lora_r "$lora_r" \
    lora.lora_alpha "$lora_alpha" \
    model.add_pos_emb "$add_pos_emb" \
    model.feat_fusion "$feat_fusion" \
    optimizer.max_grad_norm "$max_grad_norm" \
    seed "$seed" \
    model.fuse_with_id "$fuse_with_id" \
    model.llama_model_path "$llama_model_path" \
    model.use_location_token "$use_location_token"

