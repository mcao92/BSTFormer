train_data = dict(
    type="DavisAugData",
    data_root="/home/caomiao/datasets/DAVIS/DAVIS-480/JPEGImages/480p",
    in_channs=1,
)

train_mask = dict(
    type="random_real",
    frames=8,
    size_h=256,
    size_w=256,
    mask_path="mask/mask.mat",
)
