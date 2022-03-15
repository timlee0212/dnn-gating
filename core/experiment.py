import os
from tabnanny import verbose

import numpy as np
import torch

from core import model, plugin, dataset
from timm.data import resolve_data_config


class Experiment:
    def __init__(self, config):
        self.config = config
        #Single process by default
        self.main_proc = True

        self._init_plugins()
        self._set_seed()
        if config.Experiment.dist:
            self._init_dist()
        else:
            assert (self.config.Experiment.gpu_ids is None) or len(self.config.Experiment.gpu_ids)==1, "Cannot support multi-GPU in single process mode!"
            self.device = "cpu" if self.config.Experiment.gpu_ids is None else torch.device("cuda")

        self._init_model()
        #Initilize Dataset
        self._init_data()

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
        for pl_conf in self.config.Plugins:
            self.plugin_list.append(plugin.createPlugin(pl_conf.name, **pl_conf.params.__dict__))
    
    def _init_model(self):
        #Create Model
        self.model = model.createModel(**self.config.Model.__dict__)
        for plg in self.plugin_list:
            plg.modelCreationHook(self.model)
        #Load Checkpoint
        if os.path.exists(os.path.join(self.config.Experiment.path, "model_ckpt")):
            #Load model checkpoint
            pass
        for plg in self.plugin_list:
            plg.modelManipHook(model)

        self.model.to(self.device)
        if self.config.Experiment.dist:
            self.model = torch.nn.parallel.distributed.DistributedDataParallel(self.model, device_ids=self.local_rank)

    def _init_data(self):
        data_config = resolve_data_config(self.config.Data.__dict__, model=self.model, default_cfg=self.config.Data.__dict__ , verbose=self.main_proc)
        self.train_loader = dataset.createTrainLoader(self.config.Data.name, self.config, data_config)
        self.val_loader = dataset.createValLoader(self.config.Data.name, self.config, data_config)

    def _set_seed(self):
        seed = self.config.Experiment.seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    def _init_dist(self):
        """
        Initilize the distributed running environment
        """
        self.local_rank = int(os.environ["LOCAL_RANK"])
        #Indicate whether current process is the main
        self.main_proc = (self.local_rank == 0)

        self.device = "cpu" if self.config.Experiment.gpu_ids is None \
                            else torch.device("cuda", self.local_rank) 
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
