from clinicadl.utils.exceptions import ClinicaDLArgumentError


def check_gpu():
    import torch

    if not torch.cuda.is_available():
        raise ClinicaDLArgumentError(
            "No GPU is available. Please add the --no-gpu flag to run on CPU."
        )
