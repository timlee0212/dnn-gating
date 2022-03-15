#Adopted from the code in Detectron2
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
class Registry(object):

    def __init__(self, name):
        """
        Args:
            name (str): the name of this registry
        """
        self._name = name

        self._obj_map = {}

    def _do_register(self, name, obj):
        assert (
            name not in self._obj_map
        ), "An object named '{}' was already registered in '{}' registry!".format(name, self._name)
        self._obj_map[name] = obj

    def register(self, obj=None):
        """
        Register the given object under the the name `obj.__name__`.
        Can be used as either a decorator or not. See docstring of this class for usage.
        """
        if obj is None:
            # used as a decorator
            def deco(func_or_class):
                name = func_or_class.__name__
                self._do_register(name, func_or_class)
                return func_or_class

            return deco

        # used as a function call
        name = obj.__name__
        self._do_register(name, obj)

    def get(self, name):
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError("No object named '{}' found in '{}' registry!".format(name, self._name))
        return ret

_plugin_to_class = {}
_plugin_registry = Registry("Plugins")

def registerPlugin(obj):
    _plugin_registry.register(obj)
    _plugin_to_class[obj.pluginName.fget(obj)] = obj.__name__
    
def findPlugin(name):
    try:
        plugin = _plugin_registry.get(name)
    except KeyError:
        plugin = _plugin_registry.get(_plugin_to_class[name])
    return plugin


_trainer_to_class = {}
_trainer_registry = Registry("Trianers")


def registerTrainer(obj):
    _trainer_registry.register(obj)
    _trainer_to_class[obj.trainerName.fget(obj)] = obj.__name__


def findTrainer(name):
    try:
        trainer = _trainer_registry.get(name)
    except KeyError:
        trainer = _trainer_registry.get(_trainer_to_class[name])
    return trainer