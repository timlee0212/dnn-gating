from abc import ABCMeta, abstractmethod
from core.registry import findTrainer

class Trainer(metaclass=ABCMeta):
    @property
    @abstractmethod
    def trainerName(self):
        pass

    @abstractmethod
    def trainModel(self, model, **kwargs):
        pass

    @abstractmethod
    def evalModel(self, model, **kwargs):
        pass

    # @abstractmethod
    # def saveState(self, path):
    #     pass
    # @abstractmethod
    # def loadState(self, path):
    #     pass

import trainers
def createTrainer(trainer_name, **kwargs):
    """
    Load plugin based on its name. The plugin_name should match its folder
    """
    trainer_class = findTrainer(trainer_name)

    return trainer_class(**kwargs)