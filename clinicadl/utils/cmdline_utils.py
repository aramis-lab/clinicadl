from clinicadl.utils.exceptions import ClinicaDLArgumentError


def check_gpu():
    import torch

    if not torch.cuda.is_available():
        raise ClinicaDLArgumentError(
            "No GPU is available. To run on CPU, please set gpu to false or add the --no-gpu flag if you use the commandline."
        )
