## Install Instructions

### System Dependencies:
- Conda version >= 4.9
- NVIDIA driver >= 460.32
- Cuda toolkit >= 11.0
- pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html

Steps:

1. Create a new conda environment with: conda env create -f environment.yml

2. Install python bindings for isaacgym: https://developer.nvidia.com/isaac-gym

3. run the following command from this directory: pip install -e . 

### Running Example

1. run scripts/train_self_collision.py to get weights for robot self collision checking.

2. Run python franka_reacher.py, which will launch isaac gym with a franka robot trying to reach a red mug. In the isaac gym gui, search for "ee_target" and toggle "pose override", now you can move the target pose by using the sliders.

