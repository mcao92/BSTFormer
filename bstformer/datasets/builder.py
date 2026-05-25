from bstformer.utils.registry import Registry, build_from_cfg


DATASETS = Registry("datasets")


def build_dataset(cfg, default_args=None):
    return build_from_cfg(cfg, DATASETS, default_args)
