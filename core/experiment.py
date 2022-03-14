from core import model, plugin
import os

class Experiment:
    def __init__(self, config):
        self.config = config

        self._init_plugins()
        if config["Experiment"]["dist"]:
            self._init_dist()

        #Create Model
        self.model = model.createModel(**config["Experiment"]["model"])
        for plg in self.plugin_list:
            plg.modelCreationHook(self.model)
        #Load Checkpoint
        if os.path.exists(os.path.join(config["Experiment"]["path"], "model_ckpt")):
            #Load model checkpoint
            pass
        for plg in self.plugin_list:
            plg.modelManipHook(model)

        #Initilize Dataset

        #Initilize Trainer

    @classmethod
    def from_folder(cls, folder, new_path=None):
        """
        Resume an experiment from a predefined folder
        """
        pass

    def run(self):
        """
        Run experiment based on the predefined configs
        """
        pass

    def dump(self):
        """
        Dump the experiment checkpoint for use later
        """
        pass

    def _init_plugins(self):
        self.plugin_list = []
        for pl_conf in self.config['Plugins']:
            self.plugin_list.append(plugin.createPlugin(pl_conf['name'], **pl_conf['params']))