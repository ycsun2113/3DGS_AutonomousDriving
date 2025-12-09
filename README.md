<!-- # 3DGS Scene Reconstruction Methods for Autonomous Driving
Final project for ROB 535: Self-Driving Cars. -->

<p align="center">

  <h1 align="center">3D Gaussian Splatting Scene Reconstruction Methods for Autonomous Driving</h1>
  <h3 align="center">
    <strong>Hao-Yu Chan</strong>
    ,
    <strong>Peng-Chen Chen</strong>
    ,
    <strong>Yung-Ching Sun</strong>
  </h3>
  <h3 align="center"><a href="./doc/report.pdf">Report</a> | <a href="./doc/poster.pdf">Poster</a></h3>
  <div align="center"></div>
</p>

1. We benchmark three state-of-the-art 3D Gaussian Splatting–based reconstruction frameworks ([Street Gaussians](https://github.com/zju3dv/street_gaussians), [OmniRe](https://github.com/ziyc/drivestudio), [Gaussian STORM](https://github.com/NVlabs/GaussianSTORM)) on large-scale autonomous driving datasets, including [Waymo](https://waymo.com/open/) and [nuScenes](https://www.nuscenes.org/).

2. Our study presents comprehensive quantitative evaluations (PSNR, SSIM, LPIPS) and qualitative comparisons, examining geometric stability, rendering fidelity, and the handling of dynamic objects in challenging driving scenarios.

3. We provide guidelines for running these 3DGS reconstruction methods, summarize their respective strengths and limitations, and discuss potential directions for future research and system improvements.

---

# Demo

<p align="center">
  <img src="media/waymo_scene_552_compare.gif" alt="Scene reconstruction results on Waymo dataset (scene 023)" width="720"/>
</p>

# Reconstructed Results
- [Waymo Dataset scene_id 023](./media/waymo_scene_023_compare.mp4)
- [Waymo Dataset scene_id 552](./media/waymo_scene_552_compare.mp4)
- [NuScenes Dataset scene_id 000](./media/nuscenes_mini_000_compare.mp4)
- [NuScenes Dataset scene_id 003](./media/nuscenes_mini_003_compare.mp4)

---

# Getting Started

<div align="center">

## Street Gaussians: Modeling Dynamic Urban Scenes with Gaussian Splatting

[Official Street Gaussians Repository](https://github.com/zju3dv/street_gaussians) • [Project Page](https://zju3dv.github.io/street_gaussians) • [arXiv Paper](https://arxiv.org/pdf/2401.01339.pdf)

</div>

### Installation
```bash
git clone --recursive https://github.com/PCChen827/street_gaussians.git
cd submodules/street_gaussians

# create conda environment
conda create -n street-gaussian python=3.10
conda activate street-gaussian

# Load GCC and CUDA (if running on a cluster/HPC)
module load gcc/11.2.0
module load cuda/12.8.1

# install python dependencies
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt

# install submodules of street gaussians
python -m pip install --no-build-isolation ./submodules/diff-gaussian-rasterization
python -m pip install --no-build-isolation ./submodules/simple-knn
python -m pip install --no-build-isolation ./submodules/simple-waymo-open-dataset-reader

conda install -c conda-forge colmap=3.10 ceres-solver suitesparse
conda install -c conda-forge ninja cmake -y

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
python script/test_gaussian_rasterization.py
```

### Waymo Open Dataset
This guide explains how to download, preprocess, and generate necessary artifacts (LiDAR depth and sky masks) for the Waymo dataset.

### 1. Dataset Download

Follow the instructions in [EmerNeRF/NOTR.md](https://github.com/NVlabs/EmerNeRF/blob/main/docs/NOTR.md).

#### Prerequisites
1. Sign up for a [Waymo Open Dataset Account](https://waymo.com/open/).
2. Install the **gcloud SDK**.
3. Download `datasets/download_waymo.py` from the EmerNeRF repository.

#### Download Command
```bash
python datasets/download_waymo.py --target_dir ../street_gaussians/data/waymo/raw --scene_ids 23 552
```

### 2. Data Preprocessing

Follow the instructions in [Street Gaussians](https://github.com/zju3dv/street_gaussians).

#### Create Split File

Create a configuration file at script/waymo/waymo_splits/my_scenes.txt with the following content:

```Plaintext
# scene_id, seg_name, start_timestep, end_timestep, scene_type
23,seg104554,0,-1,ego-static
552,seg448767,0,-1,dynamic
```
#### Run Converter

Run waymo_converter.py to preprocess the scenes.

```bash
python script/waymo/waymo_converter.py \
    --root_dir ./data/waymo/raw \
    --save_dir ./data/waymo/processed \
    --split_file script/waymo/waymo_splits/my_scenes.txt \
    --segment_file script/waymo/waymo_splits/segment_list_train.txt
```

#### Generate LiDAR Depth
Run the following commands to generate LiDAR depth maps for each scene:

```bash
# For Scene 023
python script/waymo/generate_lidar_depth.py --datadir ./data/waymo/processed/023

# For Scene 552
python script/waymo/generate_lidar_depth.py --datadir ./data/waymo/processed/552
```

#### Generate Sky Mask
We use GroundingDINO and SAM to generate sky masks. It is recommended to create a separate conda environment for this step.

Step 1. Environment Setup

```bash
# Create and activate a new environment
conda create -n g_dino python=3.8
conda activate g_dino

# Load GCC (if running on a cluster/HPC)
module load gcc/11.2.0

# install GroundingDINO
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -e .

# install dependencies from the main project
pip install -r ../street-gaussians/requirements.txt
```

Step 2. Download Checkpoint

Ensure you have the SAM checkpoint (sam_vit_h_4b8939.pth) ready.

Step 3: Run Generation

Run the sky mask generation script for the processed scenes:

```bash
# For Scene 023
python script/waymo/generate_sky_mask.py \
    --datadir data/waymo/processed/023 \
    --sam_checkpoint ./GroundingDINO/sam_vit_h_4b8939.pth

# For Scene 552
python script/waymo/generate_sky_mask.py \
    --datadir data/waymo/processed/552 \
    --sam_checkpoint ./GroundingDINO/sam_vit_h_4b8939.pth
```

### NuScenes Dataset
This guide explains how to download, preprocess, and generate necessary artifacts (LiDAR depth and sky masks) for the Waymo dataset.

### 1. Dataset Download

Follow the Step 1 to 3 in [drivestudio/docs/NuScenes.md](https://github.com/ziyc/drivestudio/blob/main/docs/NuScenes.md).

### 2. Data Preprocessing
Convert the data to street gaussians format and generate the lidar depth
```bash
cd street_gaussians

python ./script/nuscenes/drivestudio_to_streetgaussian.py   --ds_path ./data/nuscenes/processed_10Hz/000   --out_path ./data/sg_nuscenes/000
python ./script/nuscenes/generate_lidar_depth_nuscenes.py --datadir ./data/sg_nuscenes/000

python ./script/nuscenes/drivestudio_to_streetgaussian.py   --ds_path ./data/nuscenes/processed_10Hz/003   --out_path ./data/sg_nuscenes/003
python ./script/nuscenes/generate_lidar_depth_nuscenes.py --datadir ./data/sg_nuscenes/003
```

#### Generate Sky Mask
As same as the steps in waymo preprocessing

```bash
python script/nuscenes/generate_sky_mask.py --datadir ./data/sg_nuscenes/000 --sam_checkpoint ./GroundingDINO/sam_vit_h_4b8939.pth

python script/nuscenes/generate_sky_mask.py --datadir ./data/sg_nuscenes/003 --sam_checkpoint ./GroundingDINO/sam_vit_h_4b8939.pth
```

### Configuration
Modify `default.yaml` as you need

### Training
```bash
# Load GCC and CUDA (if running on a cluster/HPC)
module load gcc/11.2.0
module load cuda/12.8.1

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# python train.py --config configs/xxxx.yaml
python train.py --config configs/waymo/waymo_scene_023.yaml
python train.py --config configs/waymo/waymo_scene_552.yaml

python train.py --config configs/nuscenes/nuscenes_mini_000.yaml
python train.py --config configs/nuscenes/nuscenes_mini_003.yaml
```
### Rendering
```bash
# Load GCC and CUDA (if running on a cluster/HPC)
module load gcc/11.2.0
module load cuda/12.8.1

# python render.py --config configs/xxxx.yaml mode {evaluate, trajectory}
# For video rendering
python render.py --config configs/waymo/waymo_scene_023.yaml mode trajectory
# For frame by frame rendering
python render.py --config configs/waymo/waymo_scene_023.yaml mode evaluate

# For video rendering
python render.py --config configs/waymo/waymo_scene_552.yaml mode trajectory
# For frame by frame rendering
python render.py --config configs/waymo/waymo_scene_552.yaml mode evaluate

# For video rendering
python render.py --config configs/nuscenes/nuscenes_mini_000.yaml mode trajectory
# For frame by frame rendering
python render.py --config configs/nuscenes/nuscenes_mini_000.yaml mode evaluate

# For video rendering
python render.py --config configs/nuscenes/nuscenes_mini_003.yaml mode trajectory
# For frame by frame rendering
python render.py --config configs/nuscenes/nuscenes_mini_003.yaml mode evaluate
```
### Evaluation
Evaluation using PNSR, SSIM, and LPIPS
```bash
# python metrics.py --config configs/xxxx.yaml
python metrics.py --config configs/waymo/waymo_scene_023.yaml
python metrics.py --config configs/waymo/waymo_scene_552.yaml

python metrics.py --config configs/nuscenes/nuscenes_mini_000.yaml
python metrics.py --config configs/nuscenes/nuscenes_mini_003.yaml
```
---
<div align="center">

## OmniRe: Omni Urban Scene Reconstruction
[Official OmniRe Repository](https://github.com/ziyc/drivestudio) • [Project Page](https://ziyc.github.io/omnire/) • [arXiv Paper](https://arxiv.org/abs/2408.16760)

</div>

### Installation
```
# Clone the repository with submodules
git clone --recursive https://github.com/ziyc/drivestudio.git
cd drivestudio

# Create the environment
conda create -n drivestudio python=3.9 -y
conda activate drivestudio
pip install -r requirements.txt
pip install git+https://github.com/nerfstudio-project/gsplat.git@v1.3.0
pip install git+https://github.com/facebookresearch/pytorch3d.git
pip install git+https://github.com/NVlabs/nvdiffrast

# Set up for SMPL Gaussians
cd third_party/smplx/
pip install -e .
cd ../..
```

### Dataset Preperation
 - To prepare the Waymo Open Dataset, please refer to [Waymo Dataset Preprocess Instruction](https://github.com/ziyc/drivestudio/blob/main/docs/Waymo.md)
 - To prepare the NuScenes mini Dataset, please refer to [NuScenes Dataset Preprocess Instruction](https://github.com/ziyc/drivestudio/blob/main/docs/NuScenes.md)
 
 Run `export PYTHONPATH=$(pwd)` in the drivestudio root folder before running the code, otherwise the preprocess scripts can't be found

 ### Training
For different dataset, adjust [dataset](https://github.com/ziyc/drivestudio/blob/main/configs/omnire.yaml#2) line in `omnire.yaml`, and the [scene_idx](https://github.com/ziyc/drivestudio/blob/main/configs/datasets/waymo/3cams.yaml#13) in the dataset you are working with. It will take approximately 5 hours and 80 GB memory to train a scene.
```
# This part is for GreatLakes
# ======================================================
export CUDA_HOME=/sw/pkgs/arc/cuda/11.7.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
# ======================================================

export PYTHONPATH=$(pwd)
start_timestep=0 # start frame index for training
end_timestep=-1 # end frame index, -1 for the last frame
output_root="results/street_gs/scene_000" # root directory to save results
project="drivestudio" # wandb project name
expname="nuscene_3cam" # experiment name

# configs/datasets/waymo/3cams.yaml

python tools/train.py \
    --config_file configs/omnire.yaml \
    --output_root $output_root \
    --project $project \
    --run_name $expname \
``` 
---

<div align="center">

## STORM: Spatio-Temporal Reconstruction Model for Large-Scale Outdoor Scenes

[Official STORM Repository](https://github.com/NVlabs/GaussianSTORM) • [Project Page](https://jiawei-yang.github.io/STORM/) • [arXiv Paper](https://arxiv.org/abs/2501.00602)

</div>

### Installation

```bash
cd submodules/GaussianSTORM

# create conda environment
conda create -n storm python=3.10 -y
conda activate storm

# install python dependencies
pip install -r requirements.txt

# install gsplat (for batch-wise rendering support)
pip install git+https://github.com/nerfstudio-project/gsplat.git@2b0de894232d21e8963179a7bbbd315f27c52c9c
#   └─ if the above fails, drop the commit hash:
#       pip install git+https://github.com/nerfstudio-project/gsplat.git
```

### Dataset Preparation
- To prepare the Waymo Open Dataset, please refer to [Waymo Data](submodules/GaussianSTORM/docs/WAYMO.md)
- They haven't provided instructions for preparing other datasets.

After completing all preprocessing steps, the dataset files should be organized as the following structure:
```bash
your_data_root
    ├── annotations
    │   └── segment-10455472356147194054_1560_000_1580_000_with_camera_labels.json # this file name will be different for each scene
    ├── datasets
    │   └── waymo
    │       ├── training
    │       │   └── scene_id
    │       │       ├── cam_to_ego      # camera to ego-vehicle transformations: {cam_id}.txt
    │       │       ├── cam_to_world    # camera to world transformations: {timestep:03d}_{cam_id}.txt
    │       │       ├── depth_flows_4   # downsampled (1/4) depth flow maps: {timestep:03d}_{cam_id}.npy
    │       │       ├── dynamic_masks   # bounding-box-generated dynamic masks: {timestep:03d}_{cam_id}.png
    │       │       ├── ego_to_world    # ego-vehicle to world transformations: {timestep:03d}.txt
    │       │       ├── ground_label_4  # downsampled (1/4) ground labels extracted from point cloud, used for flow evaluation only: {timestep:03d}.txt
    │       │       ├── images          # original camera images: {timestep:03d}_{cam_id}.jpg
    │       │       ├── images_4        # downsampled (1/4) camera images: {timestep:03d}_{cam_id}.jpg
    │       │       ├── intrinsics      # camera intrinsics: {cam_id}.txt
    │       │       ├── lidar           # lidar data: {timestep:03d}.bin
    │       │       └── sky_masks_4     # sky masks: {timestep:03d}_{cam_id}.png
    │       └── validation
    │           ├── scene_id
    │               ├── cam_to_ego
    │               ├── cam_to_world
    │               ├── depth_flows_4
    │               ├── dynamic_masks
    │               ├── ego_to_world
    │               ├── ground_label_4
    │               ├── images
    │               ├── images_4
    │               ├── intrinsics
    │               ├── lidar
    │               ├── sky_masks_4
    └── scene_list
        ├── waymo_train.txt
        └── waymo_val.txt

```

### Training

You can train the model with multi-GPU using the following command:

```bash
torchrun --nproc_per_node=2 main_storm.py \
    --project project_name \
    --exp_name exp_name \
    --data_root ../your_data_root \ # replace this with your data root.
    --batch_size 4 \
    --num_iterations 30000 --lr_sched constant \
    --model STORM-B/8 --num_motion_tokens 16 \
    --use_sky_token --use_affine_token \
    --load_depth --load_flow --load_ground \
    --enable_depth_loss --enable_flow_reg_loss --flow_reg_coeff 0.005 --enable_sky_opacity_loss \
    --enable_perceptual_loss --perceptual_loss_start_iter 5000 \
    --output_dir ./work_dir \
    --num_workers 8 \
    --ckpt_every_n_iters 1000 \
    --auto_resume 
```

Notes:
 - Checkpoints and logs are saved to `./work_dirs/<project>/<exp_name>/`
 - `batch_size` is per-GPU, global batch = batch_size * #GPUs * #nodes
 - See `main_storm.py` for additional arguments.

### Evaluation
After training, run the evaluation script to compute the PSNR, SSIM, and LPIPS metrics for your model on the dataset.

```bash
torchrun --nproc_per_node=2 main_storm.py \
    --project project_name \
    --exp_name exp_name \
    --data_root ../your_data_root \ # replace this with your data root.
    --batch_size 4 \
    --num_iterations 30000 --lr_sched constant \
    --model STORM-B/8 --num_motion_tokens 16 \
    --use_sky_token --use_affine_token \
    --load_depth --load_flow --load_ground \
    --enable_depth_loss --enable_flow_reg_loss --flow_reg_coeff 0.005 --enable_sky_opacity_loss \
    --enable_perceptual_loss --perceptual_loss_start_iter 5000 \
    --output_dir ./work_dir \
    --num_workers 8 \
    --ckpt_every_n_iters 1000 \
    --auto_resume \
    --evaluate # this parameter specifies the evaluation mode 
```

### Inference


```bash
python inference.py \
    --project project_name \
    --exp_name exp_name \
    --data_root ../storm2.3/data/STORM2 \ # replace this with your data root.
    --model STORM-B/8 --num_motion_tokens 16 \
    --use_sky_token --use_affine_token \
    --load_depth --load_flow --load_ground \
    --load_from ./work_dir/project_name/exp_name/checkpoints/latest.pth \ # replace this with the path to the checkpoint
    --output_dir ./work_dir 
```


```bash
# Visualize only the reconstructed RGB results.
python inference_rgb_only.py \
    --project project_name \
    --exp_name exp_name \
    --data_root ../storm2.3/data/STORM2 \ # replace this with your data root.
    --model STORM-B/8 --num_motion_tokens 16 \
    --use_sky_token --use_affine_token \
    --load_depth --load_flow --load_ground \
    --load_from ./work_dir/project_name/exp_name/checkpoints/latest.pth \ # replace this with the path to the checkpoint
    --output_dir ./work_dir 
```
