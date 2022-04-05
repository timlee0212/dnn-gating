from core import Inspector, Config
import torch
import logging
import argparse
from plugins.PrecisionGating.pg_ops import PGAttention
import os

class Dumper(Inspector):
    def __init__(self, config):
        super(Dumper, self).__init__(config)
        self.layer_masks = {}
        self.path = config.dump_path
        #Manipulate models to regiser hook
        #The hook is defined inside the class to write layer masks
        def _dump_mask(module, input, output):
            self.layer_masks[module.name] = module.mask
        for n, m in self.model.named_modules():
            if isinstance(m, PGAttention):
                m.register_forward_hook(_dump_mask)

    #Override the run function
    def run(self):
        self.trainer.evalModel(self.model, n_iter=1)
        torch.save(self.layer_masks, self.path)


parser = argparse.ArgumentParser(description='Training Config', add_help=True)
parser.add_argument('-c', '--config', default=None, type=str,
                    help='YAML config file specifying arguments')
parser.add_argument('-w', '--checkpoint', default=None, type=str,
                    help='Checkpoint File to be used for inspection.')
parser.add_argument('-p', '--exp-path', default=None, type=str,
                    help='Load model and checkpoint from previous experiment path')
parser.add_argument('-d', '--dump-path', default=None, type=str,
                    help='The file name for the dump mask')

__logger = logging.getLogger()
__logger.setLevel(logging.INFO)
if __name__ == "__main__":

    args = parser.parse_args()
    if args.dump_path is None:
        __logger.error("You must provide a valid output path for the dump mask.")
        exit()

    if args.config is not None:
        config = Config(args.config)
        config.dump_path = args.dump_path
        config.Experiment.resume = False
        # Config validation batchsize so that it can fit into the GPU mem
        config.Trainer.val_batch_size = config.Trainer.val_batch_size / len(config.Experiment.gpu_ids)
        if args.checkpoint is not None:
            config.Experiment.checkpoint_path = args.checkpoint
            __logger.info("Using checkpoint at {0}".format(args.checkpoint))
        else:
            config.Experiment.checkpoint_path = None
        __logger.info("Using config file. Ignoring experiment path setting")
        ins = Dumper(config)
    elif args.exp_path is not None:
        ins = Dumper.from_file(args.exp_path, dump_path=args.dump_path)
    else:
        __logger.error("No valid config file provided.")
        exit()

    ins.run()









