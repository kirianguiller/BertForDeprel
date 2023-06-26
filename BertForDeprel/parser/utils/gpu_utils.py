import torch

def get_gpus_configuration(gpu_ids):

    def get_gpu_devices(gpu_ids: str):
        gpus = []

        for gpu in gpu_ids.split(","):
            gpus.append(gpu)

        return gpus

    gpus = get_gpu_devices(gpu_ids)


    train_on_gpu = False
    multi_gpu = False
    gpu_to_train = "0"
    if torch.cuda.is_available():
        if len(gpus) == 0:
            gpu = gpus[0]
            if gpu == "-1":
                train_on_gpu = False
                gpu = "0"

            elif gpu == "-2":
                gpu = "0"
                train_on_gpu = True
                multi_gpu = True

            else:
                train_on_gpu = True
                gpu_to_train = gpu
        else:
            # multi gpus selecting is not avalaible for the moment (it will train on all gpus)
                train_on_gpu = True
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

    print(f"Train on gpu: {train_on_gpu}")
    device = torch.device(f"cuda:0" if train_on_gpu else "cpu")
    print(f"Train on device: {device}")


    return device, train_on_gpu, multi_gpu
