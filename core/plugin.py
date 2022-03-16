from abc import ABCMeta, abstractmethod
from core.registry import findPlugin
import logging

# Base class of the plugins
# Defines all necessary members for a new plugin

class Plugin(metaclass=ABCMeta):
    @property
    @abstractmethod
    def pluginName(self):
        return "Plugin"

    @abstractmethod
    def loadState(self, checkpoint_path=''):
        pass

    @abstractmethod
    def saveState(self, checkpoint_path=''):
        pass

    def modelCreationHook(self, model):
        """
        The hook to manipulate the model after creating, initializing the model.
        """
        pass

    def modelManipHook(self, model):
        """
        The hook to manipulate the model after creating, initializing, and loading the pretrained weight if any.
        """
        pass

    def epochHeadHook(self, model, epoch_id):
        """
        The hook at the head of each epoch.
        """

    def preForwardHook(self, model, inputs, targets, iter_id):
        """
        The hook to process model before the forward of each iteration.
        """
        pass

    def preBackwardHook(self, model, inputs, targets, loss, iter_id):
        """
        The hook to process model before the loss backward of each iteration.
        """
        pass

    def preUpdateHook(self, model, inputs, targets, loss, iter_id):
        """
        The hook to process the model and/or loss after the loss backward and before the weight update process.
        """
        pass

    def epochHeadHook(self, model, epoch_id):
        """
        The hook at the tail of each epoch.
        """
        pass


    def iterTailHook(self, model, inputs, targets, logger, iter_id):
        """
        The hook to log the information at the end of each iteration
        """
        pass

    def epochTailHook(self, model, epoch_id):
        """
        The hook to log information at the end of each epoch (after evaluation process)
        """
        pass

    def evalIterHook(self, model, iter_id, logger=None):
        """
        The hook to process log information at the end of each iteration of the evaluation.
        """
        pass

    def evalTailHook(self, model, logger=None):
        """
        The hook to process log information at the end of each iteration of the evaluation.
        """
        pass

#Register the modules
#Move to here to avoid circular import
import plugins


def createPlugin(plugin_name, **kwargs):
    """
    Load plugin based on its name. The plugin_name should match its folder
    """
    plugin_class = findPlugin(plugin_name)

    return plugin_class(**kwargs)

