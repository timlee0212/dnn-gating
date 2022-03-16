import argparse
from core import Config, inspector
import os
import logging


parser = argparse.ArgumentParser(description='Training Config', add_help=True)
parser.add_argument('-c', '--config', default=None, type=str,
                    help='YAML config file specifying arguments')
parser.add_argument('-w', '--checkpoint', default=None, type=str,
                    help='Checkpoint File to be used for inspection.')
parser.add_argument('-p', '--exp-path', default=None, type=str,
                    help='Load model and checkpoint from previous experiment path')

__logger = logging.getLogger()
__logger.setLevel(logging.INFO)
if __name__=="__main__":
    
    args = parser.parse_args()
    if args.config is not None:
        config = Config(args.config)
        config.Experiment.resume = False
        #Config validation batchsize so that it can fit into the GPU mem
        config.Trainer.val_batch_size = config.Trainer.val_batch_size / len(config.Experiment.gpu_ids)
        if args.checkpoint is not None:
            config.Model.checkpoint_path = args.checkpoint
            __logger.info("Using checkpoint at {0}".format(args.checkpoint))
        __logger.info("Using config file. Ignoring experiment path setting")
        ins = inspector.Inspector(config)
    elif args.exp_path is not None:
        ins = inspector.Inspector.from_folder(args.exp_path)
    else:
        __logger.error("No valid config file provided.")
        exit()
    
    ins.run()


    