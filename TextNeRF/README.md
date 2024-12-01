# Installation
The NeRF training code is based on [ngp_pl](https://github.com/kwea123/ngp_pl) project, which has strict requirements due to dependencies on other libraries, if you encounter installation problem due to hardware/software mismatch, I'm afraid there is no intention to support different platforms.

## Hardware
- OS: Ubuntu 20.04
- NVIDIA GPU with Compute Compatibility >= 75 and memory > 6GB (Tested with RTX 2080 Ti), CUDA 11.3 (might work with older version)
- 32GB RAM (in order to load full size images)

## Software
- Python>=3.8
- Python libraries
    - Install pytorch by `pip install torch==1.11.0 --extra-index-url https://download.pytorch.org/whl/cu113`
    - Install torch-scatter following their [instruction](https://github.com/rusty1s/pytorch_scatter#installation)
    - Install tinycudann following their [instruction](https://github.com/NVlabs/tiny-cuda-nn#pytorch-extension) (pytorch extension)
    - Install apex following their [instruction](https://github.com/NVIDIA/apex#linux)
    - Install core requirements by `pip install -r requirements.txt`
- Cuda extension: Upgrade `pip` to >= 22.1 and run `pip install models/csrc/` (please run this each time you `pull` the code)


# Prepare Dataset
- Colmap data
For custom data, run `colmap` and get a folder `sparse/0` under which there are `cameras.bin`, `images.bin` and `points3D.bin`.

- Annotate source text labels
To model the text regions in a scene, choose some valid views (3~5) after the sfm (colmap) has been run.
Annotate the selected view through [PPOCRLabel](https://github.com/PFCCLab/PPOCRLabel/blob/main/README.md), and you will get a file named `Label.txt` containing the labeled images and corresponding text instances. (Please note that when annotating the same text instance from different perspectives, a consistent annotation format should be used throughout. For example, the clockwise quadrilateral starting from the upper left vertex could be a standard annotation format).

# Training
- Customize the scenario-specific parameters for training by referring to the `configs/default.yaml` file. More options can be found in [hparams.py](https://github.com/cuijl-ai/TextNeRF/blob/main/TextNeRF/misc/hparams.py), feel free to modify the parameters based on specific needs.
- Run the training script `python train_nerf.py -c path/to/custom_config.yaml` to create the neural radiance field of the text scene.

# Synthesizing and auto-annotating data
- Use the `annotate_dataset.py` script to synthesize new data and annotate the corresponding text labels. For detailed usage instructions, please refer to the comments within the script.
