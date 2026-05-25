# Sparse Transformer for Ultra-Sparse Sampled Video Compressive Sensing

Miao Cao, Siming Zheng, Lishun Wang, Ziyang Chen, David Brady and Xin Yuan

IEEE Transactions on Multimedia, 2026

[[Paper]](https://doi.org/10.1109/TMM.2025.3639993)

---

> Video snapshot compressive imaging reconstructs high-speed video frames from
> a single compressed measurement. This repository provides the BSTFormer code
> for ultra-sparse sampled video compressive sensing, including training,
> grayscale simulation testing, FLOPs profiling, pretrained checkpoint loading,
> and common SCI utilities.

---

## Network Architecture

The BSTFormer model is registered as `BSTFormer` and implemented in:

```text
bstformer/models/network.py
```

The default experiment config is:

```text
configs/BSTFormer/bstformer_base.py
```

Important config fields:

```python
model = dict(type="BSTFormer", color_channels=1, units=4, dim=48)
checkpoints = "checkpoints/bstformer_base.pth"
work_dir = "work_dirs/bstformer_base"
```

## Installation

Create and activate a conda environment:

```bash
conda create -n bstformer python=3.10
conda activate bstformer
```

Install a CUDA-enabled PyTorch version that matches your GPU and CUDA runtime.
Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

The prepared local environment used during development is named `bstformer`.

## Project Layout

```text
BSTFormer/
|-- bstformer/
|   |-- datasets/        # Dataset registry and dataset implementations
|   |-- models/          # Model registry and BSTFormer network
|   `-- utils/           # Config, checkpoint, metrics, masks, logging
|-- configs/
|   |-- _base_/          # Shared runtime and data config fragments
|   `-- BSTFormer/       # BSTFormer experiment configs
|-- tools/
|   |-- train.py         # Training entry
|   |-- test.py          # Simulation testing entry
|   |-- flops.py         # FLOPs and parameter profiling
|   `-- dist_train.sh    # Multi-GPU training launcher
|-- checkpoints/         # Pretrained weights
|-- test_datasets/       # Grayscale simulation benchmark data
|-- mask/                # Mask files
`-- requirements.txt
```

## Training

Support single-GPU and multi-GPU training. First download the DAVIS training
dataset, then modify `data_root` in:

```text
configs/_base_/davis_aug.py
```

Make sure `data_root` points to your local training image directory:

```python
train_data = dict(
    type="DavisAugData",
    data_root="path/to/DAVIS/DAVIS-480/JPEGImages/480p",
    in_channs=1,
)
```

The default training mask uses the included mask file:

```python
train_mask = dict(
    type="random_real",
    frames=8,
    size_h=256,
    size_w=256,
    mask_path="mask/mask.mat",
)
```

Launch single-GPU training:

```bash
python tools/train.py configs/BSTFormer/bstformer_base.py
```

Launch multi-GPU training:

```bash
bash tools/dist_train.sh
```

The current distributed launcher is:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=3278 tools/train.py configs/BSTFormer/bstformer_base.py --distributed
```

Adjust `CUDA_VISIBLE_DEVICES` and `--nproc_per_node` for your machine.

Training outputs are saved to:

```text
work_dirs/bstformer_base/log/
work_dirs/bstformer_base/show/
work_dirs/bstformer_base/train_images/
work_dirs/bstformer_base/checkpoints/
```

Resume training from a saved checkpoint:

```bash
python tools/train.py configs/BSTFormer/bstformer_base.py --resume work_dirs/bstformer_base/checkpoints/epoch_10.pth
```

## Testing BSTFormer on Grayscale Simulation Dataset

The repository includes six grayscale simulation benchmark files under
`test_datasets/simulation/` and a pretrained checkpoint at
`checkpoints/bstformer_base.pth`.

Run simulation testing:

```bash
python tools/test.py configs/BSTFormer/bstformer_base.py
```

Or specify checkpoint weights explicitly:

```bash
python tools/test.py configs/BSTFormer/bstformer_base.py --weights checkpoints/bstformer_base.pth
```

Testing outputs are saved to:

```text
work_dirs/bstformer_base/test_log/
work_dirs/bstformer_base/test_results_vis/
```

`tools/test.py` reports final reconstruction PSNR and SSIM.

## FLOPs / Parameters

Profile FLOPs and parameter counts:

```bash
python tools/flops.py configs/BSTFormer/bstformer_base.py
```

The profiling input shape is controlled by:

```python
profile = dict(
    height=256,
    width=256,
    frames=8,
    repeat=10,
)
```

## Checkpoints and Data

- `checkpoints/bstformer_base.pth` is the default pretrained model.
- `mask/mask.mat` is the default mask file.
- `test_datasets/simulation/` contains the default grayscale simulation data.
- `work_dirs/` stores generated logs, visualizations, TensorBoard files, and
  checkpoints. It is ignored by git and can be deleted safely.

## Citation

If this code or model helps your work, please cite our paper:

```bibtex
@article{cao2026sparse,
  title={Sparse Transformer for Ultra-Sparse Sampled Video Compressive Sensing},
  author={Cao, Miao and Zheng, Siming and Wang, Lishun and Chen, Ziyang and Brady, David and Yuan, Xin},
  journal={IEEE Transactions on Multimedia},
  volume={28},
  pages={1730--1743},
  year={2026},
  doi={10.1109/TMM.2025.3639993}
}
```

Related video SCI projects:

```text
EfficientSCI++: https://github.com/mcao92/EfficientSCI-plus-plus
STFormer:      https://github.com/ucaswangls/STFormer
```
