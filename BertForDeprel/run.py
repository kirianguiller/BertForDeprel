import argparse
import os
import json
from parser.cmds import Predict, Train
from parser.utils.gpu_utils import get_gpus_configuration
from parser.utils.types import get_default_model_params
import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create the Biaffine Parser model.")
    subparsers = parser.add_subparsers(title="Commands", dest="mode")
    subcommands = {"predict": Predict(), "train": Train()}
    for name, subcommand in subcommands.items():
        subparser = subcommand.add_subparser(name, subparsers)
        subparser.add_argument(
            "--conf", "-c", help="path to config file (.json)"
        )
        subparser.add_argument(
            "--root_folder_path", "-f", help="path to models folder"
        )
        subparser.add_argument(
            "--model_name", "-m", help="name of current saved model"
        )
        subparser.add_argument(
            "--path_annotation_schema", default="", help="path to annotation schema (default : in folder/annotation_schema.json"
        )

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

    model_params = get_default_model_params()
    
    # if a conf
    if args.conf:
        if os.path.isfile(args.conf):
            with open(args.conf, "r") as infile:
                model_params = json.loads(infile.read())
        else:
            raise Exception(f"You provided a --conf parameter but no config was found in `{args.conf}`")
    

    if args.root_folder_path:
        model_params["root_folder_path"] = args.root_folder_path
    if args.model_name:
        model_params["model_name"] = args.model_name

    if "/" in model_params["model_name"]:
        raise Exception(f"`model_name` parameter has to be a filename, and not a relative or absolute path : `{model_params['model_name']}`")

    if not os.path.isdir(model_params["root_folder_path"]):
        os.makedirs(model_params["root_folder_path"])

    if args.path_annotation_schema:
        print("You provided a path to a custom annotation schema, we will use this one for your model")
        with open(args.path_annotation_schema, "r") as infile:
            model_params["annotation_schema"] = json.loads(infile.read())

    print(args.path_annotation_schema)
    print(f"Set the seed for generating random numbers to {args.seed}")
    torch.manual_seed(args.seed)
    
    args.device, args.train_on_gpu, args.multi_gpu = get_gpus_configuration(args.gpu_ids)

    print(f"Override the default configs with parsed arguments")

    print(f"Run the subcommand in mode {args.mode}")
    cmd = subcommands[args.mode]
    cmd(args, model_params)
