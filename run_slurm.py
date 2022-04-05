import argparse
import logging
from core import Config, Experiment
import os, sys

#FIX BUG
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


parser = argparse.ArgumentParser(description='Training Config', add_help=True)
parser.add_argument('-c', '--config', default=None, type=str,
                    help='YAML config file specifying arguments')
parser.add_argument('-p', '--exp-path', default=None, type=str,
                    help='Load model and checkpoint from previous experiment path')
parser.add_argument('-n', '--new-path', default=None, type=str,
                    help='YAML config file specifying arguments')
parser.add_argument('-r', '--resume', default=True, action='store_true',
                    help='Resume from previous experiment ')
parser.add_argument('-s', '--slurm-node', default="athena", type=str,
                    help="Slurm cluter name to launch task on")

if __name__=="__main__":
    args = parser.parse_args()
    assert args.config is not None or args.exp_path is not None

    logger = logging.getLogger("Launcher")
    logger.setLevel(logging.INFO)
    config = Config(args.config) if args.config is not None else \
        Config(os.path.join(args.exp_path, "config.yaml")) 

    #Check if we need bootstrap
    if "SLURM_JOB_ID" not in os.environ:
        logger.info("Bootstraping with Slrum Commands")
        n_gpus = len(config.Experiment.gpu_ids)
        command = "srun --gres=gpu:{0} --ntasks-per-node={1} --cpus-per-task={2} python run_slurm.py {3}".format(n_gpus, n_gpus, n_gpus*8, " ".join(sys.argv[1:]))
        logger.info(command)
        os.system(command)
    else:
        #Get info of the process from the environment
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank)
        if rank == 0:
            logger.info("Launching experiment with slurm, Job ID: {0}, Assigned GPUs: {1}, GPU_IDS: {2}".format(os.environ["SLRUM_JOB_ID"], os.environ["SLURM_GPUS"], os.environ["CUDA_VISIBLE_DEVICES"]))

        #Update the GPU list based on the settings
        config.Experiment.gpu_ids = [int(id) for id in os.environ["CUDA_VISIBLE_DEVICES"].split(',')]

        #Now we start the experiment
        if args.exp_path is None:
            exp = Experiment(config)
        else:
            exp = Experiment.from_folder(args.exp_path, args.new_path, args.resume)
        exp.run()






