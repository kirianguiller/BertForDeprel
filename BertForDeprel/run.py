import argparse
import json
from parser.cmds import PredictCmd, TrainCmd
from parser.cmds.cmd import CMD
from parser.utils.gpu_utils import get_devices_configuration
from parser.utils.types import ModelParams_T
from pathlib import Path
from typing import Dict

import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create the Biaffine Parser model.")
    subparsers = parser.add_subparsers(title="Commands", dest="mode")
    subcommands: Dict[str, CMD] = {"predict": PredictCmd(), "train": TrainCmd()}
    for name, subcommand in subcommands.items():
        subparser = subcommand.add_subparser(name, subparsers)
        subparser.add_argument(
            "--conf", "-c", type=Path, help="path to config file (.json)"
        )

        subparser.add_argument(
            "--gpu_ids",
            default="-2",
            help="ID of GPU to use (-1 for cpu, -2 for all gpus, 0 for gpu 0; 0,1 for "
            "gpu 0 and 1)",
        )
        subparser.add_argument("--batch_size", type=int, help="batch_size to use")
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

    model_params = ModelParams_T()

    if args.conf:
        if args.conf.is_file():
            with open(args.conf, "r") as infile:
                custom_model_params = json.loads(infile.read())
                model_params = ModelParams_T.from_dict(custom_model_params)
        else:
            raise Exception(
                "You provided a --conf parameter but no config was found in "
                f"`{args.conf}`"
            )

    # TODO
    # if model_params.get("embedding_cached_path", "") == "":
    #     model_params["embedding_cached_path"] = Path.home() / ".cache" /
    #       "huggingface"
    #     print(f"No `embedding_cached_path` provided, saving huggingface pretrained "
    #           "embedding in default cache location : "
    #           f"`{model_params['embedding_cached_path']}` ")

    if args.batch_size:
        model_params.batch_size = args.batch_size

    print(f"Set the seed for generating random numbers to {args.seed}")
    torch.manual_seed(args.seed)

    args.device_config = get_devices_configuration(args.gpu_ids)

    print("Override the default configs with parsed arguments")

    print(f"Running subcommand '{args.mode}'")
    cmd = subcommands[args.mode]
    cmd.run(args, model_params)
