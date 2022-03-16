import argparse
from core import Config, Experiment
import os

#FIX BUG
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

parser = argparse.ArgumentParser(description='Training Config', add_help=True)
parser.add_argument('-c', '--config', default=None, type=str,
                    help='YAML config file specifying arguments')
parser.add_argument('-p', '--exp-path', default=None, type=str,
                    help='Load model and checkpoint from previous experiment path')
parser.add_argument('-n', '--new-path', default=None, type=str,
                    help='YAML config file specifying arguments')
parser.add_argument('-r', '--resume', default=True, action='store_true',
                    help='Resume from previous experiment ')

if __name__=="__main__":
    args = parser.parse_args()
    assert args.config is not None or args.exp_path is not None

    config = Config(args.config) if args.config is not None else \
            Config(os.path.join(args.exp_path, "config.yaml"))

    #First check if it is a distributed environment
    dist_need_bootstrap = config.Experiment.dist and "LOCAL_RANK" not in os.environ

    bootstrap_path = args.exp_path if args.new_path is None else args.new_path
    #Check if we need to copy the folder before launch in distributed environment
    if args.new_path is not None and args.resume and dist_need_bootstrap:
        #We copy the checkpoint folder before doing bootstrap
        Experiment.from_folder(args.exp_path, args.new_path, args.resume, copy_only=True)

    if dist_need_bootstrap:
        gpu_ids = [str(x) for x in config.Experiment.gpu_ids]
        os.system("CUDA_VISIBLE_DEVICES={0} torchrun --nnodes=1 --nproc_per_node {1} "
                  "run.py {2} {3}".format(",".join(gpu_ids),
                                          str(len(gpu_ids)),
                                          " " if args.config is None else "-c "+args.config,
                                          " " if bootstrap_path is None else "-p "+ bootstrap_path))
    #Now we start the experiment
    else:
        if args.exp_path is None:
            exp = Experiment(config)
        else:
            exp = Experiment.from_folder(args.exp_path, args.new_path, args.resume)
        exp.run()






