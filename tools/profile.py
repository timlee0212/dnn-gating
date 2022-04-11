import argparse
import functools
import logging
import os
import sys
sys.path.append(os.getcwd())

import torch

from core import Inspector, Config
from timm.models.vision_transformer import Attention
from timm.models import levit
from models import pvt
from torch.utils.benchmark import Timer

_candi_list = (Attention, levit.Attention,
               levit.AttentionSubsample, pvt.Attention)


class Profiler(Inspector):
    def __init__(self, config):
        super(Profiler, self).__init__(config)
        self.attn_input = {}
        self.path = config.report_path
        # Manipulate models to regiser hook
        # The hook is defined inside the class to write layer masks

        def _dump_input(module, input, output, name):
            # Convert to bool type to save space
            self.attn_input[name] = list(input)

        for n, m in self.model.named_modules():
            if isinstance(m, _candi_list):
                m.register_forward_hook(functools.partial(_dump_input, name=n))

    # Override the run function
    def run(self, use_cpu=False):
        self.trainer.evalModel(self.model, n_iter=1)
        # Now we test the latency of each layer
        if use_cpu:
            self.model.to("cpu")
        attn_lat = {}
        for n, m in self.model.named_modules():
            if n in self.attn_input.keys():
                # Time the attention
                if not use_cpu:
                    #TODO:FIX TEMP SOLUTION
                    self.attn_input[n][0] = self.attn_input[n][0].detach()
                else:
                    self.attn_input[n][0] = self.attn_input[n][0].detach().cpu()
                attn_timer = Timer(stmt="module(*input)", globals={
                    "module": m, "input": [torch.unsqueeze(self.attn_input[n][0][0, :, :],0), *self.attn_input[n][1:]] })
                self.cmd_logger.info("Testing {0}...".format(n))
                attn_lat[n] = attn_timer.timeit(100).mean
                self.cmd_logger.info(
                    "Run {0} for 100 loops, latency: {1}ms...".format(n, attn_lat[n]*1000))

        self.cmd_logger.info("Total Attention Latency: {0}ms".format(
            sum(attn_lat.values())*1000))

        with open(os.path.join(self.path, self.config.Model.model_name+"_attn_{0}.csv".format('cpu' if use_cpu else 'gpu')), "w") as f:
            for (key, item) in attn_lat.items():
                f.write("{key}, {item}\n".format(key=key, item=item * 1000))


parser = argparse.ArgumentParser(description='Training Config', add_help=True)
parser.add_argument('-c', '--config', default=None, type=str,
                    help='YAML config file specifying arguments')
parser.add_argument('-d', '--report-path', default=None, type=str,
                    help='YAML config file specifying arguments')
parser.add_argument('--cpu', default=False, action='store_true',
                    help='Use cpu for attention latency test.')

__logger = logging.getLogger()
__logger.setLevel(logging.INFO)
if __name__ == "__main__":

    args = parser.parse_args()
    if args.report_path is None:
        __logger.error(
            "You must provide a valid output path for the report file.")
        exit()

    if args.config is not None:
        config = Config(args.config)
        config.report_path = args.report_path
        config.Experiment.resume = False
        # Remove all plugins to test vanilla model
        if hasattr(config, "Plugins"):
            config.Plugins = []
        config.Trainer.amp = False
        # Config validation batchsize so that it can fit into the GPU mem
        config.Trainer.val_batch_size = config.Trainer.val_batch_size / \
            len(config.Experiment.gpu_ids) // 2  # Since we diable the amp
        ins = Profiler(config)
    else:
        __logger.error("No valid config file provided.")
        exit()

    if not os.path.exists(args.report_path):
        os.mkdir(args.report_path)
    ins.run(use_cpu=args.cpu)
