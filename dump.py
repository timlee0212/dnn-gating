import argparse
import functools
import logging
import os

import torch
import numpy as np

from core import Inspector, Config
from plugins.PrecisionGating.pg_ops import PGAttention
from torch.utils.benchmark import Timer


class Dumper(Inspector):
    def __init__(self, config):
        super(Dumper, self).__init__(config)
        self.layer_masks = {}
        self.attn_input = {}
        self.path = config.dump_path
        # Manipulate models to regiser hook
        # The hook is defined inside the class to write layer masks

        def _dump_mask(module, input, output, name):
            # Convert to bool type to save space
            self.layer_masks[name] = module.mask.detach(
            ).cpu().numpy().astype(bool)
            self.attn_input[name] = torch.unsqueeze(
                input[0][0, :, :], 0).detach().cpu()

        for n, m in self.model.named_modules():
            if isinstance(m, PGAttention):
                m.register_forward_hook(functools.partial(_dump_mask, name=n))

    # Override the run function
    def run(self, use_cpu=False):
        self.trainer.evalModel(self.model, n_iter=1)
        np.save(os.path.join(self.path, self.model.__class__.__name__ +
                "_mask.npy"), self.layer_masks)
        # Now we test the latency of each layer
        if use_cpu:
            self.model.to("cpu")
        attn_lat = {}
        for n, m in self.model.named_modules():
            if n in self.attn_input.keys():
                # Time the attention
                if not use_cpu:
                    self.attn_input[n] = self.attn_input[n].to('cuda')
                attn_timer = Timer(stmt="module(input)", globals={
                                   "module": m, "input": self.attn_input[n]})
                self.cmd_logger.info("Testing {0}...".format(n))
                attn_lat[n] = attn_timer.timeit(100).mean
                self.cmd_logger.info(
                    "Run {0} for 100 loops, latency: {1}ms...".format(n, attn_lat[n]*1000))
        
        self.cmd_logger.info("Total Attention Latency: {0}ms".format(sum(attn_lat.values())*1000))

        with open(os.path.join(self.path, self.model.__class__.__name__+"_attn_{0}.csv".format('cpu' if use_cpu else 'gpu')), "w") as f:
            for (key, item) in attn_lat.items():
                f.write("{key}, {item}\n".format(key=key, item=item * 1000))


parser = argparse.ArgumentParser(description='Training Config', add_help=True)
parser.add_argument('-c', '--config', default=None, type=str,
                    help='YAML config file specifying arguments')
parser.add_argument('-w', '--checkpoint', default=None, type=str,
                    help='Checkpoint File to be used for inspection.')
parser.add_argument('-p', '--exp-path', default=None, type=str,
                    help='Load model and checkpoint from previous experiment path')
parser.add_argument('-d', '--dump-path', default=None, type=str,
                    help='The path for dump result')
parser.add_argument('--cpu', default=False, action='store_true',
                    help='Use cpu for attention latency test.')

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
    ins.run(use_cpu=args.cpu)
