# 3DGS Scene Reconstruction Methods for Autonomous Driving
Final project for ROB 535: Self-Driving Cars.
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