import argparse
import os
import pathlib
from parser.cmds import Evaluate, Predict, Train
from parser.config import Config
from parser.utils.os_utils import path_or_name
from parser.utils.gpu_utils import get_gpus_configuration

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
        subparser.add_argument(
            "--path_annotation_schema", default="", help="path to annotation schema (default : in folder/annotation_schema.json"
        )

        # subparser.add_argument('--preprocess', '-p', action='store_true',
        #                        help='whether to preprocess the data first')
        subparser.add_argument('--gpu_ids', default='-2',
                               help='ID of GPU to use (-1 for cpu, -2 for all gpus, 0 for gpu 0; 0,1 for gpu 0 and 1)')
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


    args = parser.parse_args()

    # if not os.path.isdir(args.folder):
    #     os.makedirs(args.folder)

    path_models = os.path.join(args.folder, "models")
    if not os.path.isdir(path_models):
        os.makedirs(path_models)

    if path_or_name(args.model) == "name":
        args.model = os.path.join(args.folder, "models", args.model)

    if not args.path_annotation_schema:
        args.path_annotation_schema = os.path.join(args.folder, "annotation_schema.json")
    print(args.path_annotation_schema)
    # else:
    #     args.model = os.path.join(args.models, args.model)
    print("args.model", args.model)
    # print(f"Set the max num of threads to {args.threads}")
    print(f"Set the seed for generating random numbers to {args.seed}")
    # print(f"Set the device with ID {args.device} visible")
    # torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    
    args.device, args.train_on_gpu, args.multi_gpu = get_gpus_configuration(args.gpu_ids)

    print(f"Override the default configs with parsed arguments")
    path_file_directory = pathlib.Path(__file__).parent.absolute()
    path_config = os.path.join(path_file_directory, args.conf)
    print(path_config)
    print("KK ", Config)
    args = Config(path_config).update(vars(args))
    args.patience = 30
    args.epochs = 300
    print(args)

    print(f"Run the subcommand in mode {args.mode}")
    cmd = subcommands[args.mode]
    cmd(args)
