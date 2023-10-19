from dataclasses import dataclass

import torch.backends.mps
import torch.cuda


@dataclass
class DeviceConfig:
    device: torch.device
    multi_gpu: bool


def get_devices_configuration(gpu_ids):
    gpus = gpu_ids.split(",")

    use_gpu = False
    multi_gpu = False
    if torch.cuda.is_available():
        if len(gpus) == 1:
            gpu = gpus[0]
            if gpu == "-1":
                use_gpu = False
                gpu = "0"

            elif gpu == "-2":
                gpu = "0"
                use_gpu = True
                multi_gpu = True
            else:
                use_gpu = True
        else:
            # multi gpus selecting is not avalaible for the moment (it will train on
            # all gpus)
            use_gpu = True
            multi_gpu = True

        # Number of gpus
        if multi_gpu:
            gpu_count = torch.cuda.device_count()
            print(f"{gpu_count} gpus detected.")
            if gpu_count > 1:
                multi_gpu = True
            else:
                multi_gpu = False
        else:
            multi_gpu = False

    print(f"Using gpu: {use_gpu}")
    if use_gpu:
        device = torch.device("cuda:0")
    # TODO: probably should be user-configurable whether MPS is used
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    return DeviceConfig(device, multi_gpu)
