import inspect

import torch

from bstformer.utils.registry import Registry, build_from_cfg


OPTIMIZERS = Registry("optimizers")


def _register_torch_optimizers():
    for name in dir(torch.optim):
        if name.startswith("_"):
            continue
        optimizer = getattr(torch.optim, name)
        if inspect.isclass(optimizer) and issubclass(optimizer, torch.optim.Optimizer):
            if OPTIMIZERS.get(name) is None:
                OPTIMIZERS.register_module(optimizer)


_register_torch_optimizers()


def build_optimizer(cfg, default_args=None):
    return build_from_cfg(cfg, OPTIMIZERS, default_args)
