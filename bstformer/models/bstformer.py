from .network import BSTFormer as _BSTFormer

from .builder import MODELS


@MODELS.register_module
class BSTFormer(_BSTFormer):
    """Registry wrapper for the BSTFormer implementation."""

    pass
