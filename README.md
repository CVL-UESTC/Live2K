# Live2K

[![arXiv](https://img.shields.io/badge/arXiv-2607.04151-b31b1b.svg?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2607.04151)

Official implementation of **Perceiving Better Moments: Cover Frame Reselection and Enhancement for Live Photos with the Live2K Dataset**, accepted to **ECCV 2026**.

Live Photos contain a high-quality cover image and a short burst of video frames. In real smartphones, these two parts are produced by different imaging pipelines: the cover image receives full computational photography processing, while the video frames are usually lower-resolution, compressed, and less color-consistent. When users choose another video frame as the cover, the selected frame often looks much worse than the original cover.

<p align="center">
  <img src="imgs/figure1.png" alt="Live Photo imaging pipelines" width="80%">
</p>
<p align="center"><em>Live Photo imaging pipelines for high-quality cover photos and lower-quality video frames.</em></p>

This repository studies **Live Photo Cover Frame Reselection and Enhancement (LPRE)**. Given a user-reselected low-quality frame, its adjacent video frames, and the original high-quality cover image as reference, the goal is to reconstruct a high-quality replacement cover frame with improved detail, color, and dynamic range.

<p align="center">
  <img src="imgs/figure2.png" alt="Live Photo cover frame reselection and enhancement" width="100%">
</p>
<p align="center"><em>LPRE improves a directly reselected cover frame with reference-guided enhancement.</em></p>

## Highlights

- We formulate the **LPRE** task for Live Photo cover frame reselection and enhancement.
- We build **Live2K**, a real-world dataset containing **2,042 Live Photos**.
- We provide a unified one-stage baseline with:
  - multi-frame temporal fusion,
  - cover-guided color and appearance enhancement,
  - reference-guided super-resolution.

## Dataset

<p align="center">
  <img src="imgs/figure3.png" alt="Live2K dataset examples" width="100%">
</p>
<p align="center"><em>Example scenes from the Live2K dataset.</em></p>

The original Live2K dataset can be downloaded from Baidu Netdisk:

[Download Live2K Dataset](https://pan.baidu.com/s/1gMonGcy5Nnt7ES4Ps6LQIQ?pwd=sprq)

Extraction code:

```text
sprq
```

Full dataset will be release soon.

The dataset loader expects each Live Photo sample to be stored as one subfolder:

```text
Live2K_root/
├── 000001/
│   ├── gt.png
│   ├── ref.png
│   └── lq_sequence/
│       ├── 000.png
│       ├── 001.png
│       ├── ...
│       └── 008.png
├── 000002/
│   ├── gt.png
│   ├── ref.png
│   └── lq_sequence/
│       ├── 000.png
│       ├── ...
│       └── 008.png
└── ...
```

Where:

- `gt.png`: high-quality target frame.
- `ref.png`: original high-quality cover image used as the reference.
- `lq_sequence/`: nine low-quality adjacent video frames. The code sorts all `.png` files in this folder, so filenames should sort in temporal order.

After preparing the data, update the `dataroot` fields in:

```text
options/train/Apple/train_sr_tsa.yml
options/train/OPPO/train_sr_tsa.yml
options/test/Apple/test.yml
options/test/OPPO/test.yml
```

## Repository Structure

```text
Live2K/
├── data/                         # Dataset and dataloader code
├── models/
│   ├── archs/                    # LPENet and network components
│   ├── losses/                   # Training losses
│   ├── train_model.py            # Training model wrapper
│   └── test_model.py             # Testing model wrapper
├── options/
│   ├── train/Apple/train_sr_tsa.yml
│   ├── train/OPPO/train_sr_tsa.yml
│   ├── test/Apple/test.yml
│   ├── test/OPPO/test.yml
│   └── test/test_speed.yml
├── pretrained/                   # Put pretrained weights here
├── train.py
├── test.py
├── test_speed.py
├── train.sh
├── test.sh
└── requirement.txt
```

## Installation

Create a clean environment:

```bash
conda create -n live2k python=3.11 -y
conda activate live2k
```

Install PyTorch and torchvision according to your CUDA version. For example:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Then install the remaining dependencies:

```bash
pip install -r requirement.txt
```

Note: the default training configs use `optim_g.type: Muon`. The current development environment uses a standalone `muon.py` module that has no pip package metadata. To train with Muon, install or vendor a compatible `MuonWithAuxAdam` implementation. Otherwise, change `optim_g.type` in the training config to `Adam`.

## Pretrained Models

Put pretrained weights in `pretrained/`. The provided test configs expect:

```text
pretrained/iPhone.pth
pretrained/oppo.pth
```

You can change `path.pretrain_network_g` in the corresponding test config if your weights are stored elsewhere.

## Training

Train on the Apple split:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
  --use-env --nproc_per_node=2 --master_port=1145 train.py \
  -opt options/train/Apple/train_sr_tsa.yml --launcher pytorch
```

Train on the OPPO split:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
  --use-env --nproc_per_node=2 --master_port=1145 train.py \
  -opt options/train/OPPO/train_sr_tsa.yml --launcher pytorch
```

Training outputs are saved under:

```text
checkpoint/experiments/<experiment_name>/
```

## Testing

Test the OPPO model:

```bash
CUDA_VISIBLE_DEVICES=0 python test.py -opt options/test/OPPO/test.yml
```

Test the Apple model:

```bash
CUDA_VISIBLE_DEVICES=0 python test.py -opt options/test/Apple/test.yml
```

Run speed evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 python test_speed.py -opt options/test/test_speed.yml
```

Results are saved under:

```text
results/results/<experiment_name>/
```

## Citation

If this project is useful for your research, please cite:

```bibtex
@misc{lou2026perceiving,
  title  = {Perceiving Better Moments: Cover Frame Reselection and Enhancement for Live Photos with the Live2K Dataset},
  author = {Lou, Junyu and Chen, Kai and You, Weiyi and Zeng, Hui and Zhang, Lei and Gu, Shuhang},
  year   = {2026},
  note   = {Project page: https://github.com/CVL-UESTC/Live2K}
}
```

## Keywords

Live Photo, image enhancement, image super-resolution, cover frame reselection, Live2K.
