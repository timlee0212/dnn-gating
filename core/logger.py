
from tensorboardX import SummaryWriter
import os
import torch

class logger:
    def __init__(self, config):
        self.batch_size = config["trainer"]["batch_size"]
        self.steps_per_epoch = config["trainer"]["steps_per_epoch"]
        self.input_size = config["data"]["input_size"]
        self.writer = SummaryWriter(os.path.join(config["experiment"]["path"], "logs"))
    
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
        pseudo_input = torch.randn(self.input_size)
        temp_model = model.clone().to("cpu")
        try:
            self.writer.add_graph(temp_model, pseudo_input)
            del temp_model
        except:
            print("Warning: Some modules of the model may not support JIT thus cannot be logged.")

    def __del__(self):
        """
        Make sure the buffer is flushed before exit
        """
        self.writer.close()
        