import inspect

import torch

from bstformer.utils.registry import Registry, build_from_cfg


LOSSES = Registry("losses")


def _register_torch_losses():
    for name, loss in inspect.getmembers(torch.nn.modules.loss):
        if name.startswith("_"):
            continue
        if inspect.isclass(loss):
            if LOSSES.get(name) is None:
                LOSSES.register_module(loss)


_register_torch_losses()


def build_loss(cfg, default_args=None):
    return build_from_cfg(cfg, LOSSES, default_args)
