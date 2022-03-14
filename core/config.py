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
        self.config_dict = yaml.safe_load(open(config_path, "r"))
        super(Config, self).__init__(self.config_dict)

    def dump(self, dump_path):
        yaml.safe_dump(self.config_dict, open(dump_path, "w"))
