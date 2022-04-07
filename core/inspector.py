import logging
import os

import numpy as np
import torch
import yaml
from timm.data import resolve_data_config
from timm.models import resume_checkpoint
from core.config import Config

from core.dataset import createValLoader
from core.model import createModel
from core.trainer import createTrainer
from core.plugin import createPlugin


class Inspector:
    def __init__(self, config):
        self.config = config

        #We only use one GPU for evaluation
        self.config.Experiment.dist = False
        self.checkpoint_path = os.path.join(config.Experiment.path, config.Experiment.exp_id, "ckpt", "last.pth.tar")

        if not hasattr(config.Experiment, "checkpoint_path") and config.Experiment.resume:
            config.Experiment.checkpoint_path = self.checkpoint_path
        # Setup output logger
        root_logger = logging.getLogger()
        log_level = logging.DEBUG if config.Experiment.debug else logging.INFO
        logging.basicConfig(level=log_level)

        self.cmd_logger = logging.getLogger("Experiment")

        # Initilize plugins
        self._init_plugins()
        # Initilize distributed environment
        self._set_seed()
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initilize Model
        self._init_model()

        # Initilize Dataset
        self._init_data()
        # Initilize Trainer
        self._init_trainer()

        # Resume Checkpoint
        self.config.Trainer.start_epoch = 0
        if self.config.Experiment.resume:
            self.cmd_logger.info("Loading Checkpoint from {0}".format(self.checkpoint_path))
            self.config.Trainer.start_epoch = resume_checkpoint(self.model, self.checkpoint_path,
                                                                optimizer=None, loss_scaler=None, log_info=True)

    @classmethod
    def from_folder(cls, folder, **kwargs):
        """
        Resume an experiment from a predefined folder
        """
        config = Config(os.path.join(folder, "config.yaml"))
        #Load the checkpoint from experiment folder by default
        config.Experiment.resume = True
        #Config validation batchsize so that it can fit into the GPU mem
        config.Trainer.val_batch_size = config.Trainer.val_batch_size / len(config.Experiment.gpu_ids)
        #Additional Configs from subclass, currently we only support one level of attributes
        for (key, val) in kwargs.items():
            config.__setattr__(key, val)
        return cls(config)

    def run(self):
        """
        Run inspector based on the predefined configs
        """
        self.trainer.evalModel(self.model)

    def _init_plugins(self):
        self.plugin_list = []
        if self.config.Plugins is not None:
            for pl_conf in self.config.Plugins:
                self.cmd_logger.debug(
                    "Initializing plugin {0}\n Params {1}".format(pl_conf.name, str(pl_conf.params.__dict__)))
                self.plugin_list.append(createPlugin(pl_conf.name, **pl_conf.params.__dict__))

    def _init_model(self):
        # Create Model
        self.model = createModel(**self.config.Model.__dict__)
        for plg in self.plugin_list:
            plg.modelCreationHook(self.model)

        if self.config.Experiment.checkpoint_path is not None:
            if os.path.exists(self.config.Experiment.checkpoint_path):
                ckpt = torch.load(self.config.Experiment.checkpoint_path, map_location="cpu")
                #Whether it is a checkpoint saved by saver
                if "state_dict" in ckpt.keys():
                    #Use resume checkpoint function
                    self.config.Experiment.resume = True
                    self.checkpoint_path = self.config.Experiment.checkpoint_path
                else:
                    #Do not need further resume since we already load the model
                    self.config.Experiment.resume = False
                    self.model.load_state_dict(ckpt)
            else:
                self.cmd_logger.warning(
                    "Cannot find the required checkpoint path {0} for model"
                        .format(self.config.Model.checkpoint_path))

        for plg in self.plugin_list:
            plg.modelManipHook(self.model)

        self.model.to(self.device, memory_format=torch.channels_last
        if self.config.Experiment.channel_last else torch.contiguous_format)

    def _init_data(self):
        data_config = resolve_data_config(self.config.Data.__dict__, model=self.model,
                                          default_cfg=self.config.Data.__dict__, verbose=True)
        self.cmd_logger.debug(data_config)
        self.val_loader, self.val_loss = createValLoader(self.config.Data.name, self.config, data_config)
        self.val_loss.to(self.device)

        self.config.Data.input_size = data_config["input_size"]

    def _init_trainer(self):
        self.trainer = createTrainer(self.config.Trainer.name, config=self.config, optimizer=None,
                                             scheduler=None, logger=None, saver=None,
                                             plugins=self.plugin_list, verbose=True, device=self.device, train_loss=None,
                                             eval_loss=self.val_loss, train_loader=None, test_loader=self.val_loader)

    def _set_seed(self):
        seed = self.config.Experiment.seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
