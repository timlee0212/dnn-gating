import argparse
import functools
import logging
import os
import sys
sys.path.append(os.getcwd())

import torch
import numpy as np

from core import Inspector, Config
from plugins.PrecisionGating.pg_ops import PGAttention
from torch.utils.benchmark import Timer


class Dumper(Inspector):
    def __init__(self, config):
        super(Dumper, self).__init__(config)
        self.layer_masks = {}
        self.attn_log = {}
        self.attn_input = {}
        self.path = config.dump_path
        # Manipulate models to regiser hook
        # The hook is defined inside the class to write layer masks

        def _dump_mask(module, input, output, name):
            # Convert to bool type to save space
            self.attn_log[name]['mask'] = module.mask.detach(
            ).cpu().numpy().astype(bool)
        
        def _dump_linear_shape(module, input, output, name, block_name):
            self.attn_log[block_name][name] = {
                "in_shape" : input[0].shape[1:],
                "out_shape" : output[0].shape[1:]
            }

        #We first get all 
        for n, m in self.model.named_modules():
            #This is hardcoded for vit-based models
            if hasattr(m, "attn") and hasattr(m, "mlp"):
                self.attn_log[n] = {}
                m.attn.register_forward_hook(functools.partial(_dump_mask, name=n))

                m.mlp.fc1.register_forward_hook(functools.partial(_dump_linear_shape, name="ffn.fc1", block_name=n))
                m.mlp.fc2.register_forward_hook(functools.partial(_dump_linear_shape, name="ffn.fc2", block_name=n))

                #Hardcoded Rule for our candidate models
                ln_name = ['q', 'k', 'v', 'qk', 'kv', 'qv', 'qkv', 'proj']
                for name in ln_name:
                    if hasattr(m.attn, name):
                        getattr(m.attn, name).register_forward_hook(functools.partial(_dump_linear_shape, name="attn."+name, block_name=n))


    # Override the run function
    def run(self):
        self.trainer.evalModel(self.model, n_iter=1)
        np.save(os.path.join(self.path, self.config.Experiment.exp_id +
                ".npy"), self.attn_log)
        import matplotlib.pyplot as plt
        for layer in self.attn_log.values():
            for head in range(layer['mask'].shape[1]):
                image = layer['mask'][0,head,:,:]
                plt.imshow(image)
                plt.show()
  


parser = argparse.ArgumentParser(description='Training Config', add_help=True)
parser.add_argument('-c', '--config', default=None, type=str,
                    help='YAML config file specifying arguments')
parser.add_argument('-w', '--checkpoint', default=None, type=str,
                    help='Checkpoint File to be used for inspection.')
parser.add_argument('-p', '--exp-path', default=None, type=str,
                    help='Load model and checkpoint from previous experiment path')
parser.add_argument('-d', '--dump-path', default=None, type=str,
                    help='The path for dump result')

__logger = logging.getLogger()
__logger.setLevel(logging.INFO)
if __name__ == "__main__":

    args = parser.parse_args()
    if args.dump_path is None:
        __logger.error(
            "You must provide a valid output path for the dump mask.")
        exit()

    if args.config is not None:
        config = Config(args.config)
        config.dump_path = args.dump_path
        config.Experiment.resume = False
        # Config validation batchsize so that it can fit into the GPU mem
        config.Trainer.val_batch_size = config.Trainer.val_batch_size / \
            len(config.Experiment.gpu_ids)
        if args.checkpoint is not None:
            config.Experiment.checkpoint_path = args.checkpoint
            __logger.info("Using checkpoint at {0}".format(args.checkpoint))
        else:
            config.Experiment.checkpoint_path = None
        __logger.info("Using config file. Ignoring experiment path setting")
        ins = Dumper(config)
    elif args.exp_path is not None:
        ins = Dumper.from_folder(args.exp_path, dump_path=args.dump_path)
    else:
        __logger.error("No valid config file provided.")
        exit()

    if not os.path.exists(args.dump_path):
        os.mkdir(args.dump_path)
    ins.run()
