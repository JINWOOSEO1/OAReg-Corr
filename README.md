<p align="center">
  <p align="center">
    <h1 align="center">OAReg (ICLR 2025)</h1>
  </p>
  <p align="center" style="font-size:16px">
    <a target="_blank" href="https://zikai1.github.io/"><strong>Mingyang Zhao</strong></a>
    ·
    <a target="_blank" href="https://scholar.google.com/citations?user=5hti_r0AAAAJ"><strong>Gaofeng Meng</strong></a>
   ·
    <a target="_blank" href="https://sites.google.com/site/yandongming/"><strong>Dong-Ming Yan</strong></a>
  </p>

![](./fig/ICLR_Teaser.png)
### [Project Page](https://zikai1.github.io/pub/CluReg/index.html) | [Paper](https://arxiv.org/pdf/2502.10704v1) | [Poster](https://zikai1.github.io/slides/ICLR25_poster.pdf)
This repository contains the official implementation of our ICLR 2025 paper "Occlusion-aware Non-Rigid Point Cloud Registration via Unsupervised Neural Deformation Correntropy".

**Towards Non-rigid or Deformable Registration of Point Clouds and Surfaces**

**Please give a star if you find this repo useful 🤡**


## Pipeline Overview
This fork wraps the OAReg core (`test_OAR.py`) with an end-to-end pipeline that
goes from a MuJoCo scene to a deformed source point cloud and dense
source ↔ target correspondences:

1. **[src/RGBD_image_generator.py](src/RGBD_image_generator.py)** &mdash; render
   multi-view RGB-D images of the source and target scenes from MuJoCo
   (`data/images/source.xml`, `data/images/target.xml`) using three cameras
   (`cam1`/`cam2`/`cam3`). Outputs per-view RGB `.jpg`, depth `.npy`, and the
   fused intrinsics/extrinsics in `data/images/camera_params.npz`.
2. **[src/build_segmented_pointclouds.py](src/build_segmented_pointclouds.py)**
   &mdash; interactively segment the object on one reference view per side with
   SAM2 (foreground/background clicks), propagate those clicks to the other
   views via depth unprojection + reprojection, run SAM2 automatically on the
   remaining views, fuse the masked per-view clouds in the world frame, and
   save `data/source/source.ply`, `data/target/target.ply`, plus paired
   world-frame click anchors in `data/images/user_correspondence.npy`.
3. **[src/test_OAR.py](src/test_OAR.py)** &mdash; OAReg's deformation learning.
   Normalizes the source/target clouds, lifts the user-picked anchors into the
   same normalized frame, optimizes a SIREN deformation field with LLR + MCC
   chamfer + user-correspondence hinge losses, and writes the deformed source
   to `data/save_deformed/deformed.ply` (de-normalized to the target frame).
4. **[src/find_correspondence.py](src/find_correspondence.py)** &mdash;
   recover dense source ↔ target index pairs by mutual nearest neighbour
   between the deformed source and the target, gated by an adaptive distance
   threshold. Saves `(M, 2)` int pairs to `data/correspondence/correspondence.npy`.

A helper [src/visualize_ply.py](src/visualize_ply.py) renders any combination
of the source / target / deformed clouds and the resulting correspondences
with Open3D.

### Data Layout
```
data/
  images/                # produced by step 1 + user config
    source.xml           # MuJoCo scene for the source object (user-provided)
    target.xml           # MuJoCo scene for the target object (user-provided)
    config.json          # rgb/depth path lists for source & target (user-provided)
    camera_params.npz    # cam{1,2,3}_intrinsic / cam{1,2,3}_extrinsic
    source_cam{1,2,3}.{jpg,npy}
    target_cam{1,2,3}.{jpg,npy}
    user_correspondence.npy   # (N, 2, 3) paired SAM2 click anchors
  source/source.ply      # fused source point cloud (XYZ + RGB)
  target/target.ply      # fused target point cloud (XYZ + RGB)
  save_deformed/deformed.ply
  correspondence/correspondence.npy
```


## Implementation
### 1. Prerequisites ###
The code is based on PyTorch implementation, and tested on the following environment dependencies:
```
- Linux (tested on Ubuntu 22.04.1)
- Python 3.9.19
- torch=='1.12.1+cu113'
```

In addition to the OAReg core, the new pipeline scripts depend on:
- [`mujoco`](https://mujoco.org/) (for `RGBD_image_generator.py`)
- [`opencv-python`](https://pypi.org/project/opencv-python/) and
  [`Pillow`](https://pypi.org/project/Pillow/) (for `build_segmented_pointclouds.py`)
- [SAM2](https://github.com/facebookresearch/sam2) checkpoints + repo (for the
  interactive segmentation step)
- `scipy` (for `find_correspondence.py`'s KD-tree)

### 2. Setup ###
We recommend using ```Miniconda``` to set up the environment.

#### 2.1 Create conda environment ####
```
- conda create -n oar python=3.9
- conda activate oar
```

#### 2.2 Install packages ####
```
- pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
- conda install -c fvcore -c iopath -c conda-forge fvcore iopath
- conda install pytorch3d
```


If you want the torch version match the pytorch3d version, please use ```conda list``` to check the corresponding Version, and then re-setup the torch, such as
```
- pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```

Finally, setup other libraries:
```
- pip install -r requirements.txt
- pip install mujoco opencv-python Pillow
```

For step 2 you also need a working SAM2 checkout. Clone the upstream
[facebookresearch/sam2](https://github.com/facebookresearch/sam2) repo, download
a checkpoint (we default to `sam2.1_hiera_base_plus.pt` with config
`configs/sam2.1/sam2.1_hiera_b+.yaml`), and pass the paths via
`--sam2-repo` / `--sam2-ckpt` / `--sam2-config` (see the defaults at the top
of [src/build_segmented_pointclouds.py](src/build_segmented_pointclouds.py)
&mdash; they point to a local install and should be overridden for your
machine).

### 3. Prepare the MuJoCo scenes and view config ###
Place your source/target MuJoCo XMLs at `data/images/source.xml` and
`data/images/target.xml`. Each scene must define three cameras named `cam1`,
`cam2`, `cam3`. Then create `data/images/config.json` listing the rgb/depth
paths for both sides, e.g.:
```json
{
  "camera_params": "data/images/camera_params.npz",
  "source": {
    "rgb":   ["data/images/source_cam1.jpg", "data/images/source_cam2.jpg", "data/images/source_cam3.jpg"],
    "depth": ["data/images/source_cam1.npy", "data/images/source_cam2.npy", "data/images/source_cam3.npy"]
  },
  "target": {
    "rgb":   ["data/images/target_cam1.jpg", "data/images/target_cam2.jpg", "data/images/target_cam3.jpg"],
    "depth": ["data/images/target_cam1.npy", "data/images/target_cam2.npy", "data/images/target_cam3.npy"]
  }
}
```

### 4. Run the pipeline ###
End-to-end via the helper script (uses the `python` on `PATH`; override with
`PYTHON=<path>`):
```
bash src/run_pipeline.sh
```

Or run each step manually:
```
# 1) render multi-view RGB-D for source and target
python src/RGBD_image_generator.py
python src/RGBD_image_generator.py --is-target

# 2) SAM2-segment + fuse colored point clouds (interactive on the reference view)
python src/build_segmented_pointclouds.py \
    --sam2-repo  /path/to/sam2 \
    --sam2-ckpt  /path/to/sam2.1_hiera_base_plus.pt \
    --sam2-config configs/sam2.1/sam2.1_hiera_b+.yaml

# 3) OAReg deformation learning
cd src && python test_OAR.py && cd ..

# 4) dense source<->target correspondences via mutual NN
python src/find_correspondence.py
```

During step 2 the reference view (cam1 by default; override with `--ref-cam`)
opens an OpenCV window. Controls:

| Key / mouse        | Action                                |
|--------------------|---------------------------------------|
| Left click         | add a foreground prompt               |
| Right click        | add a background prompt               |
| `p`                | re-preview the SAM2 mask              |
| `u`                | undo the last point                   |
| `r`                | reset all points                      |
| `Enter`            | confirm the mask and finalize         |
| `q`                | cancel and exit                       |

Click the **same number of foreground points** on the source and target
reference views &mdash; they are paired by click order to form the
`user_correspondence.npy` anchors consumed by `test_OAR.py`.

### 5. Visualize results ###
```
# Overlay source (red) / target (blue) / deformed (green)
python src/visualize_ply.py --mode overlay

# Lines between source and target via the recovered correspondences
python src/visualize_ply.py --mode correspondence

# Lines between deformed and target
python src/visualize_ply.py --mode deformed-correspondence
```

The deformed point clouds are saved in `data/save_deformed/` and the dense
correspondence pairs in `data/correspondence/`.


## Contact
If you have any problem, please contact us via <migyangz@gmail.com>. We greatly appreciate everyone's feedback and insights. Please do not hesitate to get in touch!

## Citation
Please give a citation of our work if you find it useful:

```bibtex
@inproceedings{zhao2025oareg,
  title={Occlusion-aware Non-Rigid Point Cloud Registration via Unsupervised Neural Deformation Correntropy},
  author={Mingyang Zhao, Gaofeng Meng, Dong-Ming Yan},
  booktitle={International Conference on Learning Representations},
  year={2025}
}
```
## Acknowledgements
Our work is inspired by several outstanding prior works, including [DPF](https://github.com/sergeyprokudin/dpf), [NSFP](https://github.com/Lilac-Lee/Neural_Scene_Flow_Prior), [NDP](https://github.com/rabbityl/DeformationPyramid), and others. We would like to acknowledge and express our deep appreciation to the authors of these remarkable contributions.


## License
OAReg is under AGPL-3.0, so any downstream solution and products (including cloud services) that include OAReg code inside it should be open-sourced to comply with the AGPL conditions. For learning purposes only and not for commercial use. If you want to use it for commercial purposes, please contact us first.
