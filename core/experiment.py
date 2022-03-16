import os

import numpy as np
import shutil
import torch
import logging
import datetime
import yaml

from core import model, plugin, dataset, logger, trainer
from timm.data import resolve_data_config
from timm.models import convert_splitbn_model, resume_checkpoint
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler
from timm.utils import CheckpointSaver


class Experiment:
    def __init__(self, config):
        self.config = config
        if not os.path.exists(config.Experiment.path):
            os.mkdir(config.Experiment.path)
        if not os.path.exists(os.path.join(config.Experiment.path, config.Experiment.exp_id)):
            os.mkdir(os.path.join(config.Experiment.path, config.Experiment.exp_id))
        if not os.path.exists(os.path.join(config.Experiment.path, "config.yaml")):
            yaml.safe_dump(config.config_dict, open(os.path.join(config.Experiment.path, config.Experiment.exp_id, "config.yaml"), 'w'))
        self.checkpoint_path = os.path.join(config.Experiment.path,  config.Experiment.exp_id, "ckpt")

        if not os.path.exists(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)
        self.config.Experiment.resume = False

        # Setup output logger
        root_logger = logging.getLogger()
        log_level = logging.DEBUG if config.Experiment.debug else logging.INFO
        logging.basicConfig(level=log_level)
        fh = logging.FileHandler(os.path.join(config.Experiment.path, config.Experiment.exp_id, datetime.datetime.now().strftime("%y%m%d-%H-%M")
                                 + config.Experiment.exp_id + ".log"))
        fh.setLevel(log_level)
        # ch = logging.StreamHandler()
        # ch.setLevel(log_level)
        # root_logger.addHandler(ch)
        root_logger.addHandler(fh)

        self.cmd_logger = logging.getLogger("Experiment")

        # Single process by default
        self.main_proc = True

        # Initilize plugins
        self._init_plugins()

        # Initilize distributed environment
        self._set_seed()
        if config.Experiment.dist:
            self._init_dist()
        else:
            assert (self.config.Experiment.gpu_ids is None) or len(
                self.config.Experiment.gpu_ids) == 1, "Cannot support multi-GPU in single process mode!"
            self.device = "cpu" if self.config.Experiment.gpu_ids is None else "cuda"
            os.environ["CUDA_VISIBLE_DEVICES"] = self.config.Experiment.gpu_ids[0]

        # Initilize Model
        self._init_model()

        # Initilize Dataset
        self._init_data()
        if self.main_proc:
            self.logger = logger.Logger(config, self.device)
            self.logger.log_model(self.model)
        else:
            self.logger = None

        # Initilize Trainer
        self._init_trainer()

        #Resume Checkpoint
        self.config.Trainer.start_epoch = 0
        if self.config.Experiment.resume:
            self.config.Trainer.start_epoch = resume_checkpoint(self.model, self.checkpoint_path, optimizer=self.optimizer,
                                             loss_scaler=self.trainer.loss_scaler, log_info=self.main_proc)

    @classmethod
    def from_folder(cls, folder, new_path=None, copy_checkpoint=False, copy_only=False):
        """
        Resume an experiment from a predefined folder
        """
        config = yaml.safe_load(os.path.join(folder, "config.yaml"))
        if new_path is not None and copy_checkpoint:
            logger.getLogger().info("Copying File to the new experiment folder")
            shutil.copytree(folder, new_path)
            config.Experiment.path = new_path
            yaml.dump(config, open(os.path.join(new_path, "config.yaml"), 'w'))
        return None if copy_only else cls(config)


    def run(self):
        """
        Run experiment based on the predefined configs
        """
        self.trainer.trainModel(self.model)

    def _init_plugins(self):
        self.plugin_list = []
        for pl_conf in self.config.Plugins:
            if self.main_proc:
                self.cmd_logger.debug(
                    "Initializing plugin {0}\n Params {1}".format(pl_conf.name, str(pl_conf.params.__dict__)))
            self.plugin_list.append(plugin.createPlugin(pl_conf.name, **pl_conf.params.__dict__))

    def _init_model(self):
        # Create Model
        self.model = model.createModel(**self.config.Model.__dict__)
        for plg in self.plugin_list:
            plg.modelCreationHook(self.model)

        if self.config.Model.checkpoint_path is not None:
            if os.path.exists(self.config.Model.checkpoint_path):
                self.model.load_state_dict(torch.load(self.config.Model.checkpoint_path, map_location="cpu"))
            else:
                if self.main_proc:
                    self.cmd_logger.warning(
                        "Cannot find the required checkpoint path {0} for model"
                        .format(self.config.Model.checkpoint_path))

        for plg in self.plugin_list:
            plg.modelManipHook(model)

        self.model.to(self.device, memory_format=torch.channels_last
                        if self.config.Experiment.channel_last else torch.contiguous_format)
        if self.config.Experiment.dist:
            self.model = torch.nn.parallel.distributed.DistributedDataParallel(self.model, device_ids=[self.local_rank,])

    def _init_data(self):
        data_config = resolve_data_config(self.config.Data.__dict__, model=self.model,
                                          default_cfg=self.config.Data.__dict__, verbose=self.main_proc)
        self.cmd_logger.debug(data_config)
        self.train_loader, self.train_loss = dataset.createTrainLoader(self.config.Data.name, self.config, data_config)
        self.val_loader, self.val_loss = dataset.createValLoader(self.config.Data.name, self.config, data_config)

        self.train_loss.to(self.device)
        self.val_loss.to(self.device)

        self.config.Data.input_size = data_config["input_size"]
        self.config.Trainer.steps_per_epoch = len(self.train_loader)

    def _init_trainer(self):
        def optimizer_kwargs(opt_cfg):
            """ cfg/argparse to kwargs helper
            Convert optimizer args in argparse args or cfg like object to keyword args for updated create fn.
            """
            kwargs = dict(
                lr=opt_cfg.lr,
                weight_decay=opt_cfg.weight_decay,
                momentum=opt_cfg.momentum)
            if getattr(opt_cfg, 'eps', None) is not None:
                kwargs['eps'] = opt_cfg.eps
            if getattr(opt_cfg, 'betas', None) is not None:
                kwargs['betas'] = opt_cfg.betas
            if getattr(opt_cfg, 'opt_args', None) is not None:
                kwargs.update(opt_cfg.opt_args)
            return kwargs

        #TODO: Separate vision and possible language task
        self.optimizer = create_optimizer_v2(self.model, opt=self.config.Trainer.opt.name, **optimizer_kwargs(self.config.Trainer.opt.params))
        #Try to be compatible with timm API
        #Urgh, I hate this dirty interface
        sched_cfg = self.config.Trainer.sched
        sched_cfg.sched = sched_cfg.name
        sched_cfg.epochs = self.config.Trainer.epochs
        sched_cfg.lr = self.config.Trainer.opt.params.lr

        self.scheduler, self.config.Trainer.scheduled_epochs = create_scheduler(sched_cfg, self.optimizer)
        if self.main_proc:
            self.cmd_logger.info("Scheduled Epochs: {0}".format(self.config.Trainer.scheduled_epochs))


        #Only the main process save checkpoints
        self.saver = None
        if self.main_proc:
            self.saver = CheckpointSaver(model=self.model, optimizer=self.optimizer, args=None,
                                     checkpoint_dir=self.checkpoint_path, recovery_dir=self.checkpoint_path,
                                     max_history=self.config.Experiment.checkpoint_hist)

        self.trainer = trainer.createTrainer(self.config.Trainer.name, config = self.config, optimizer = self.optimizer,
                                             scheduler = self.scheduler, logger = self.logger, saver = self.saver,
                                             verbose=self.main_proc, device=self.device, train_loss = self.train_loss, eval_loss = self.val_loss, train_loader=self.train_loader, test_loader=self.val_loader)


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
        # Indicate whether current process is the main
        self.main_proc = (self.local_rank == 0)

        self.device = "cpu" if self.config.Experiment.gpu_ids is None \
            else "cuda"
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

        #Set visible devices for this process
        torch.cuda.set_device(torch.device("cuda", self.local_rank))

        self.cmd_logger.info('Training in distributed mode with multiple processes, 1 GPU per process. Process {0},'
                             ' total {1}.'.format(self.local_rank, torch.distributed.get_world_size()))

        # Deal with sync BN in the distributed setting
        if self.config.Experiment.sync_bn:
            assert not self.config.Experiment.split_bn
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

            if self.main_proc:
                self.cmd_logger.warning("Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using \
                                        zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.")
        if self.config.Experiment.split_bn:
            assert self.config.Data.augs.aug_splits > 1 or self.config.Data.augs.random_earse.resplit
            self.model = convert_splitbn_model(model, max(self.config.Experiment.split_bn, 2))
