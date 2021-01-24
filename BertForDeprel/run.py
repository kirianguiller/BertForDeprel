import argparse
import os

import pathlib

from parser.cmds import Evaluate, Predict, Train
from parser.config import Config
from parser.utils.os_utils import path_or_name

import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create the Biaffine Parser model.")
    subparsers = parser.add_subparsers(title="Commands", dest="mode")
    subcommands = {"evaluate": Evaluate(), "predict": Predict(), "train": Train()}
    for name, subcommand in subcommands.items():
        print(name, subcommand)
        subparser = subcommand.add_subparser(name, subparsers)
        subparser.add_argument(
            "--conf", "-c", default="config.ini", help="path to config file"
        )
        subparser.add_argument(
            "--folder", "-f", required=True, help="path to project folder"
        )
        subparser.add_argument(
            "--model", "-m", default="model.pt", help="name of saved model"
        )

        # subparser.add_argument('--preprocess', '-p', action='store_true',
        #                        help='whether to preprocess the data first')
        # subparser.add_argument('--device', '-d', default='-1',
        #                        help='ID of GPU to use')
        subparser.add_argument(
            "--num_workers", default=8, type=int, help="Number of worker"
        )
        subparser.add_argument(
            "--seed",
            "-s",
            default=42,
            type=int,
            help="seed for generating random numbers",
        )
        subparser.add_argument(
            "--bert_type",
            "-b",
            default="bert",
            help="bert type to use (bert/camembert/mbert)",
        )
        # subparser.add_argument('--threads', '-t', default=16, type=int,
        #                        help='max num of threads')
        # subparser.add_argument('--tree', action='store_true',
        #                        help='whether to ensure well-formedness')
        # subparser.add_argument('--feat', default='tag',
        #                        choices=['tag', 'char', 'bert'],
        #                        help='choices of additional features')
    args = parser.parse_args()

    # if not os.path.isdir(args.folder):
    #     os.makedirs(args.folder)

    path_models = os.path.join(args.folder, "models")
    if not os.path.isdir(path_models):
        os.makedirs(path_models)

    if path_or_name(args.model) == "name":
        args.model = os.path.join(args.folder, "models", args.model)

    # else:
    #     args.model = os.path.join(args.models, args.model)

    print("args.model", args.model)
    # print(f"Set the max num of threads to {args.threads}")
    print(f"Set the seed for generating random numbers to {args.seed}")
    # print(f"Set the device with ID {args.device} visible")
    # torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    # Whether to train on a gpu
    train_on_gpu = torch.cuda.is_available()
    print(f"Train on gpu: {train_on_gpu}")
    args.train_on_gpu = train_on_gpu

    # Number of gpus
    if train_on_gpu:
        gpu_count = torch.cuda.device_count()
        print(f"{gpu_count} gpus detected.")
        if gpu_count > 1:
            multi_gpu = True
        else:
            multi_gpu = False
    else:
        multi_gpu = None

    args.multi_gpu = multi_gpu
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Override the default configs with parsed arguments")
    path_file_directory = pathlib.Path(__file__).parent.absolute()
    args = Config(os.path.join(path_file_directory, args.conf)).update(vars(args))
    print(args)
    # args = Config({"a test": [1,2,3,4], "other test": "avdz"}).update(vars(args))
    # print(args)
    print(f"Run the subcommand in mode {args.mode}")
    cmd = subcommands[args.mode]
    cmd(args)