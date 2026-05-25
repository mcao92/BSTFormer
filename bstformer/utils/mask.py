from .common import generate_masks, random_inp_mask, random_masks, random_real_masks


def build_mask(cfg):
    cfg = dict(cfg)
    mask_type = cfg.pop("type")
    if mask_type == "random_inp":
        return random_inp_mask(**cfg)
    if mask_type == "random":
        return random_masks(**cfg)
    if mask_type == "random_real":
        return random_real_masks(**cfg)
    if mask_type == "file":
        mask_path = cfg.pop("mask_path")
        return generate_masks(mask_path)
    raise KeyError(f"Unknown mask type: {mask_type}")
