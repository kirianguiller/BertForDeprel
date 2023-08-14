import argparse
from parser.cmds import PredictCmd, TrainCmd
from parser.cmds.cmd import CMD
from parser.utils.gpu_utils import get_devices_configuration
from parser.utils.types import ModelParams_T
from typing import Dict

import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create the Biaffine Parser model.")
    subparsers = parser.add_subparsers(title="Commands", dest="mode")
    subcommands: Dict[str, CMD] = {"predict": PredictCmd(), "train": TrainCmd()}
    for name, subcommand in subcommands.items():
        subparser = subcommand.add_subparser(name, subparsers)
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

    if args.batch_size:
        model_params.batch_size = args.batch_size

    print(f"Set the seed for generating random numbers to {args.seed}")
    torch.manual_seed(args.seed)

    args.device_config = get_devices_configuration(args.gpu_ids)

    print("Override the default configs with parsed arguments")

    print(f"Running subcommand '{args.mode}'")
    cmd = subcommands[args.mode]
    cmd.run(args)
