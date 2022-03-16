#TODO: Set a default config
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except:
    from yaml import Loader, Dumper

# Struct to convert dict to class
# Source: https://stackoverflow.com/questions/1305532/convert-nested-python-dict-to-object
class Struct(object):
    def __init__(self, data):
        for name, value in data.items():
            setattr(self, name, self._wrap(value))

    def _wrap(self, value):
        if isinstance(value, (tuple, list, set, frozenset)): 
            return type(value)([self._wrap(v) for v in value])
        else:
            return Struct(value) if isinstance(value, dict) else value


class Config(Struct):
    def __init__(self, config_path):
        default_config = yaml.safe_load(open("./configs/default.yaml", 'r'))
        new_config = yaml.safe_load(open(config_path, "r"))

        self._overwrite_dict(new_config, default_config)
        self.config_dict = default_config

        super(Config, self).__init__(self.config_dict)

    def dump(self, dump_path):
        yaml.safe_dump(self.config_dict, open(dump_path, "w"))

    def _overwrite_dict(self, src, dst):
        for key, value in src.items():
            if key in dst.keys():
                if isinstance(value, dict) and isinstance(dst[key], dict):
                    self._overwrite_dict(value, dst[key])
                else:
                    dst[key] = value
            else:
                dst[key] = value
                