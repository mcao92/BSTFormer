_base_ = [
    "../_base_/default_runtime.py",
    "../_base_/simulation_data.py",
    "../_base_/davis_aug.py",
]

model = dict(
    type="BSTFormer",
    color_channels=1,
    units=4,
    dim=48,
)

checkpoints = "checkpoints/bstformer_base.pth"

work_dir = "work_dirs/bstformer_base"

profile = dict(
    height=256,
    width=256,
    frames=8,
    repeat=10,
)

eval = dict(
    flag=True,
    interval=1,
)
