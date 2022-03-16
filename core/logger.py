
from torch.utils.tensorboard import SummaryWriter
import os
import torch
import logging

class Logger:
    def __init__(self, config):
        self.batch_size = config.Trainer.batch_size
        self.steps_per_epoch = config.Trainer.steps_per_epoch
        self.input_size = config.Data.input_size
        self.writer = SummaryWriter(os.path.join(config.Experiment.path, config.Experiment.exp_id, "logs", flush_secs=30))
        self.cmd_logger = logging.getLogger("Logger")
    
    def log_scalar(self, value, name, stage, epoch, steps=0):
        """
        Log a scalar value.
        """
        #Verify test or training
        timing = epoch if steps==0 else \
                epoch * self.steps_per_epoch + steps
        self.writer.add_scalar(stage+"/"+name, value, timing)
        
    def log_model(self, model):
        """
        Log the model structure
        """
        pseudo_input = torch.randn(self.input_size).to(self.device)
        # temp_model = model.clone().to("cpu")
        try:
            self.writer.add_graph(model, pseudo_input)
        except:
            self.cmd_logger.warning("Some modules of the model may not support JIT thus cannot be logged.")

    def __del__(self):
        """
        Make sure the buffer is flushed before exit
        """
        self.writer.close()
        
