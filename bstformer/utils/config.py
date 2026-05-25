import ast
import importlib.util
import os.path as osp


BASE_KEY = "_base_"


class Config(dict):
    @staticmethod
    def fromfile(filename):
        cfg_dict = Config._file2dict(filename)
        return Config._convert(cfg_dict)

    @staticmethod
    def _file2dict(filename):
        filename = osp.abspath(osp.expanduser(filename))
        if not osp.isfile(filename):
            raise FileNotFoundError(f"config file does not exist: {filename}")

        with open(filename, "r", encoding="utf-8") as f:
            content = f.read()
        ast.parse(content)

        module_name = osp.splitext(osp.basename(filename))[0]
        spec = importlib.util.spec_from_file_location(module_name, filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        cfg_dict = {
            key: value
            for key, value in module.__dict__.items()
            if not key.startswith("__")
        }
        if BASE_KEY not in cfg_dict:
            return cfg_dict

        base_files = cfg_dict.pop(BASE_KEY)
        if isinstance(base_files, str):
            base_files = [base_files]

        merged = {}
        cfg_dir = osp.dirname(filename)
        for base_file in base_files:
            base_dict = Config._file2dict(osp.join(cfg_dir, base_file))
            duplicated = merged.keys() & base_dict.keys()
            if duplicated:
                raise KeyError(f"Duplicate keys among base configs: {duplicated}")
            merged.update(base_dict)
        return Config._merge(cfg_dict, merged)

    @staticmethod
    def _merge(child, base):
        merged = dict(base)
        for key, value in child.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = Config._merge(value, merged[key])
            else:
                merged[key] = value
        return merged

    @staticmethod
    def _convert(value):
        if isinstance(value, dict):
            return Config({key: Config._convert(val) for key, val in value.items()})
        if isinstance(value, list):
            return [Config._convert(item) for item in value]
        return value

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = Config._convert(value)
