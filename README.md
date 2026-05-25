# BSTFormer

Config-driven BSTFormer codebase for video snapshot compressive imaging.

The project is organized in a STFormer-style layout: experiment settings live in
`configs/`, executable entry points live in `tools/`, and reusable code lives in
the `bstformer/` package.

## Project Layout

```text
BSTFormer/
|-- bstformer/
|   |-- datasets/        # Dataset registry and dataset implementations
|   |-- models/          # Model registry and BSTFormer network
|   `-- utils/           # Config, registry, checkpoint, metrics, masks, logging
|-- configs/
|   |-- _base_/          # Shared runtime/data config fragments
|   `-- BSTFormer/       # BSTFormer experiment configs
|-- tools/
|   |-- train.py         # Training entry
|   |-- test.py          # Simulation testing entry
|   |-- flops.py         # FLOPs / parameter profiling
|   `-- dist_train.sh    # Multi-GPU training launcher
|-- checkpoints/         # Pretrained weights
|-- test_datasets/       # Simulation benchmark data
|-- mask/                # Optional file-mask data
`-- requirements.txt
```

## Environment

Use the prepared conda environment:

```powershell
conda activate bstformer
```

Install dependencies if needed:

```powershell
pip install -r requirements.txt
```

The project expects PyTorch with CUDA for GPU runs.

## Main Config

Default experiment config:

```text
configs/BSTFormer/bstformer_base.py
```

Important fields:

```python
model = dict(type="BSTFormer", color_channels=1, units=4, dim=48)
checkpoints = "checkpoints/bstformer_base.pth"
work_dir = "work_dirs/bstformer_base"
```

Shared defaults are inherited from:

```text
configs/_base_/default_runtime.py
configs/_base_/simulation_data.py
configs/_base_/davis_aug.py
```

## Simulation Test

Run the six grayscale simulation benchmarks:

```powershell
python tools/test.py configs/BSTFormer/bstformer_base.py
```

Outputs are written under:

```text
work_dirs/bstformer_base/test_log/
work_dirs/bstformer_base/test_results_vis/
```

Expected output includes `psnr_mean` and `ssim_mean`.

## Training

Before training, update the DAVIS path in:

```text
configs/_base_/davis_aug.py
```

Specifically:

```python
train_data = dict(
    data_root="path/to/DAVIS/DAVIS-480/JPEGImages/480p",
)
```

The default training mask config currently expects:

```python
train_mask = dict(
    type="random_real",
    frames=8,
    size_h=256,
    size_w=256,
    mask_path="mask/mask.mat",
)
```

The repository includes `mask/mask.mat`; change `train_mask.mask_path` only if
you want to use another training mask.

Single-GPU training:

```powershell
python tools/train.py configs/BSTFormer/bstformer_base.py
```

Training outputs are written under:

```text
work_dirs/bstformer_base/log/
work_dirs/bstformer_base/show/
work_dirs/bstformer_base/train_images/
work_dirs/bstformer_base/checkpoints/
```

Saved checkpoints use this format:

```python
{
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optim_state_dict": optimizer.state_dict(),
}
```

Resume training:

```powershell
python tools/train.py configs/BSTFormer/bstformer_base.py --resume work_dirs/bstformer_base/checkpoints/epoch_10.pth
```

## Multi-GPU Training

Linux/bash launcher:

```bash
bash tools/dist_train.sh
```

Current launcher:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=3278 tools/train.py configs/BSTFormer/bstformer_base.py --distributed
```

Adjust `CUDA_VISIBLE_DEVICES` and `--nproc_per_node` for your machine.

## FLOPs / Parameters

```powershell
python tools/flops.py configs/BSTFormer/bstformer_base.py
```

The input shape is controlled by:

```python
profile = dict(
    height=256,
    width=256,
    frames=8,
    repeat=10,
)
```

## Work Directory

`work_dirs/` contains generated experiment outputs. It is safe to delete when
you want a clean workspace; it will be recreated automatically by training or
testing.

## Notes

- `tools/test.py` reports final reconstruction PSNR/SSIM, not PSNR gain.
- `checkpoints/bstformer_base.pth` is the default pretrained model for simulation testing.
- `test_datasets/simulation/` contains the default six benchmark `.mat` files.
- The model registry name is `BSTFormer`; set `model.type="BSTFormer"` in configs.
