# 3DGS Scene Reconstruction Methods for Autonomous Driving
Final project for ROB 535: Self-Driving Cars.

## Street Gaussians: Modeling Dynamic Urban Scenes with Gaussian Splatting

[Official Street Gaussians Repository](https://github.com/zju3dv/street_gaussians) • [Project Page](https://zju3dv.github.io/street_gaussians) • [arXiv Paper](https://arxiv.org/pdf/2401.01339.pdf)

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
# Reconstructed Results
- [Waymo Dataset scene_id 023](./Videos/waymo_scene_023_compare.mp4)
- [Waymo Dataset scene_id 552](./Videos/waymo_scene_552_compare.mp4)
- [NuScenes Dataset scene_id 000](./Videos/nuscenes_mini_000_compare.mp4)
- [NuScenes Dataset scene_id 003](./Videos/nuscenes_mini_003_compare.mp4)

## OmniRe: Omni Urban Scene Reconstruction
[Official OmniRe Repository](https://github.com/ziyc/drivestudio) • [Project Page](https://ziyc.github.io/omnire/) • [arXiv Paper](https://arxiv.org/abs/2408.16760)

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
## STORM: Spatio-Temporal Reconstruction Model for Large-Scale Outdoor Scenes

[Official STORM Repository](https://github.com/NVlabs/GaussianSTORM) • [Project Page](https://jiawei-yang.github.io/STORM/) • [arXiv Paper](https://arxiv.org/abs/2501.00602)
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

### Training

```bash
torchrun --nproc_per_node=2 main_storm.py \
    --project 1207_storm_023 \
    --exp_name 1207_pixel_storm_023 \
    --data_root ../storm2.3/data/STORM2 \ # replace this with your data root.
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

### Evaluation
```bash
torchrun --nproc_per_node=2 main_storm.py \
    --project 1207_storm_023 \
    --exp_name 1207_pixel_storm_023 \
    --data_root ../storm2.3/data/STORM2 \ # replace this with your data root.
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
    --project 1207_storm_023 \
    --exp_name 1207_storm_023_inference \
    --data_root ../storm2.3/data/STORM2 \ # replace this with your data root.
    --model STORM-B/8 --num_motion_tokens 16 \
    --use_sky_token --use_affine_token \
    --load_depth --load_flow --load_ground \
    --load_from ./work_dir/1207_storm_023/1207_storm_023_inference/checkpoints/latest.pth \
    --output_dir ./work_dir 

# Visualize only the reconstructed RGB results.
python inference_rgb_only.py \
    --project 1207_storm_023 \
    --exp_name 1207_storm_023_inference \
    --data_root ../storm2.3/data/STORM2 \ # replace this with your data root.
    --model STORM-B/8 --num_motion_tokens 16 \
    --use_sky_token --use_affine_token \
    --load_depth --load_flow --load_ground \
    --load_from ./work_dir/1207_storm_023/1207_storm_023_inference/checkpoints/latest.pth \
    --output_dir ./work_dir 
```
