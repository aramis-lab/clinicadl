from clinicadl.utils.exceptions import ArgumentError


def check_gpu():
    import torch

    if not torch.cuda.is_available():
        raise ArgumentError(
            "No GPU is available. Please add the --no-gpu flag to run on CPU."
        )
