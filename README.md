# 3DGS Scene Reconstruction Methods for Autonomous Driving
Final project for ROB 535: Self-Driving Cars.


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