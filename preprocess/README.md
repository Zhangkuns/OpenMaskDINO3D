## Skip the data preparation

- We’ve provided all the prepared data in [Google Drive](https://drive.google.com/drive/folders/12RV0N_mPPjD8oDWz_kIfORCY9GvJX2WN?usp=sharing). Simply download the files and place them in the annotations/ directory. You’ll then be ready to run and test the code.

## Prepare data

- Download the ScanNet dataset by following the [ScanNet instructions](https://github.com/ScanNet/ScanNet).

- Extract object masks using a pretrained 3D detector:
    - Use [Mask3D](https://github.com/JonasSchult/Mask3D) for instance segmentation. We used the [checkpoint](https://omnomnom.vision.rwth-aachen.de/data/mask3d/checkpoints/scannet200/scannet200_val.ckpt) pretrained on ScanNet200.
    - The complete predicted results (especially the masks) for the train/validation sets are too large to share (~40GB). We’ve shared the post-processed [results](https://drive.google.com/file/d/1jwQYJvkWwRmawZvNOSy6U0lnqnEiasNX/view?usp=sharing):
        - Unzip the `mask3d_inst_seg.tar.gz` file.
        - Each file under `mask3d_inst_seg` contains the predicted results for a single scene, including a list of segmented instances with their labels and segmented indices.

- Process object masks and prepare annotations:
    - We use the downloaded `mask3d_inst_seg` directly, set `segment_result_dir` to None and set `inst_seg_dir` to the path of `mask3d_inst_seg`.
    - Run: `bash preprocess/run_prepare.sh`
    - For this project, we use the following command to prepare the data:
    - For mask_ins_data:
      ```shell
      python ./preprocess/prepare_mask3d_data.py --scannet_dir "/workspace/data/scannet" --output_dir "/workspace/data/scannet/ChatScene/mask3d_data" --segment_dir "" --inst_seg_dir "/workspace/data/scannet/ChatScene/mask3d_inst_seg" --class_label_file "annotations/scannet/scannetv2-labels.combined.tsv" --apply_global_alignment --num_workers 16 --parallel
      ```
    - For scannet_mask3d_attribute
      ```shell
      python preprocess/prepare_scannet_mask3d_attributes.py --scan_dir "/workspace/data/scannet/ChatScene/mask3d_ins_data" --segmentor "mask3d" --max_inst_num 100 --version ""
      ```
    - For scanrefer_mask3d_train.json
      ```shell
      python preprocess/prepare_scanrefer_annos.py --segmentor "mask3d" --version "" --train_iou_thres "0.5" --max_obj_num "100"
      ```
    - For other data, the same with the preprocess in run_prepare.sh.

- Extract 3D features using a pretrained 3D encoder:
    - Follow [Uni3D](https://github.com/baaivision/Uni3D?tab=readme-ov-file) to extract 3D features for each instance. We used the pretrained model [uni3d-g](https://huggingface.co/BAAI/Uni3D/blob/main/modelzoo/uni3d-g/model.pt).
    - We've also provided modified code for feature extraction in this forked [repository](https://github.com/ZzZZCHS/Uni3D). Set the `data_dir` [here](https://github.com/ZzZZCHS/Uni3D/blob/main/main.py#L620) to the path to `${processed_data_dir}/pcd_all` (`processed_data_dir` is an intermediate directory set in `run_prepare.sh`). After preparing the environment, run `bash scripts/inference.sh`.

- Extract 2D features using a pretrained 2D encoder:

    - We followed [OpenScene](https://github.com/pengsongyou/openscene)'s code to calculate the mapping between 3D points and 2D image pixels. This allows each object to be projected onto multi-view images. Based on the projected masks on the images, we extract and merge DINOv2 features from multi-view images for each object.
