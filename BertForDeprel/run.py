import argparse
import json
import os
from parser.cmds import Predict, Train
from parser.cmds.cmd import CMD
from parser.utils.gpu_utils import get_devices_configuration
from parser.utils.types import AnnotationSchema_T, ModelParams_T
from typing import Dict

import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create the Biaffine Parser model.")
    subparsers = parser.add_subparsers(title="Commands", dest="mode")
    subcommands: Dict[str, CMD] = {"predict": Predict(), "train": Train()}
    for name, subcommand in subcommands.items():
        subparser = subcommand.add_subparser(name, subparsers)
        subparser.add_argument("--conf", "-c", help="path to config file (.json)")

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
        if os.path.isfile(args.conf):
            with open(args.conf, "r") as infile:
                custom_model_params = json.loads(infile.read())
                # TODO: check if the config file is valid first
                model_params.__dict__.update(custom_model_params)
                if "annotation_schema" in custom_model_params:
                    annotation_schema = AnnotationSchema_T()
                    # TODO: check if the annotation schema is valid first
                    annotation_schema.__dict__.update(
                        custom_model_params["annotation_schema"]
                    )
                    model_params.annotation_schema = annotation_schema
        else:
            raise Exception(
                "You provided a --conf parameter but no config was found in "
                f"`{args.conf}`"
            )

    # TODO
    # if model_params.get("embedding_cached_path", "") == "":
    #     model_params["embedding_cached_path"] = str(Path.home() / ".cache" /
    #       "huggingface")
    #     print(f"No `embedding_cached_path` provided, saving huggingface pretrained "
    #           "embedding in default cache location : "
    #           f"`{model_params['embedding_cached_path']}` ")

    if args.batch_size:
        model_params.batch_size = args.batch_size

    print(f"Set the seed for generating random numbers to {args.seed}")
    torch.manual_seed(args.seed)

    args.device, args.train_on_gpu, args.multi_gpu = get_devices_configuration(
        args.gpu_ids
    )

    print("Override the default configs with parsed arguments")

    print(f"Running subcommand '{args.mode}'")
    cmd = subcommands[args.mode]
    cmd(args, model_params)
