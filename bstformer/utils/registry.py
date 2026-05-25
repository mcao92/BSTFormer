import inspect


class Registry:
    def __init__(self, name):
        self.name = name
        self.module_dict = {}

    def get(self, key):
        return self.module_dict.get(key)

    def register_module(self, cls):
        module_name = cls.__name__
        if module_name in self.module_dict:
            raise KeyError(f"{module_name} is already registered in {self.name}")
        self.module_dict[module_name] = cls
        return cls


def build_from_cfg(cfg, registry, default_args=None):
    if not isinstance(cfg, dict) or "type" not in cfg:
        raise TypeError("cfg must be a dict containing key 'type'")

    args = dict(cfg)
    obj_type = args.pop("type")
    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(f"{obj_type} is not registered in {registry.name}")
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(f"type must be a str or class, got {type(obj_type)}")

    if default_args is not None:
        for key, value in default_args.items():
            args.setdefault(key, value)
    return obj_cls(**args)
