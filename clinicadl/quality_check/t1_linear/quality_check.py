"""
This file contains all methods needed to perform the quality check procedure after t1-linear preprocessing.
"""

from logging import getLogger
from pathlib import Path

import pandas as pd
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from clinicadl.generate.generate_utils import load_and_check_tsv
from clinicadl.utils.caps_dataset.data import CapsDataset
from clinicadl.utils.clinica_utils import RemoteFileStructure, fetch_file
from clinicadl.utils.exceptions import ClinicaDLArgumentError

from .models import resnet_darq_qc_18 as darq_r18
from .models import resnet_deep_qc_18 as deep_r18
from .models import squeezenet_qc as darq_sq101
from .utils import QCDataset

logger = getLogger("clinicadl.quality-check")


def quality_check(
    caps_dir: Path,
    output_path: Path,
    tsv_path: Path = None,
    threshold: float = 0.5,
    batch_size: int = 1,
    n_proc: int = 0,
    gpu: bool = True,
    amp: bool = False,
    network: str = "darq",
    use_tensor: bool = False,
    use_uncropped_image: bool = True,
):
    """
    Performs t1-linear quality-check

    Parameters
    -----------
    caps_dir: str (Path)
        Path to the input caps directory
    output_path: str (Path)
        Path to the output TSV file.
    tsv_path: str (Path)
        Path to the participant.tsv if the option was added.
    threshold: float
        Threshold that indicates whether the image passes the quality check.
    batch_size: int
    n_proc: int
    gpu: int
    amp: bool
        If enabled, uses Automatic Mixed Precision (requires GPU usage).
    network: str
        Architecture of the pretrained network pretrained network that learned to classify images that are adequately registered.
        To chose between "darq" and "deep-qc"
    use_tensor: bool
        To use tensor instead of nifti images
    use_uncropped_image: bool
        To use uncropped images instead of the cropped ones.

    """

    logger = getLogger("clinicadl.quality_check")

    if not output_path.suffix == ".tsv":
        raise ClinicaDLArgumentError(f"Output path {output_path} must be a TSV file.")

    # Fetch QC model
    home = Path.home()

    cache_clinicadl = home / ".cache" / "clinicadl" / "models"
    url_aramis = "https://aramislab.paris.inria.fr/files/data/models/dl/qc/"

    cache_clinicadl.mkdir(parents=True, exist_ok=True)

    if network == "deep_qc":
        FILE1 = RemoteFileStructure(
            filename="resnet18.pth.tar",
            url=url_aramis,
            checksum="a97a781be3820b06424fe891ec405c78b87ad51a27b6b81614dbdb996ce60104",
        )
        model = deep_r18()

    if network == "darq":
        FILE1 = RemoteFileStructure(
            filename="resnet_18_darq.pth",
            url=url_aramis,
            checksum="321928e0532f1be7a8dd7f5d805b747c7147ff52594f77ffed0858ab19c5df03",
        )

        model = darq_r18()

    if network == "sq101":
        FILE1 = RemoteFileStructure(
            filename="sq101_darq.pth",
            url=url_aramis,
            checksum="1f4f3ebd20aaa726d634165a89df12461d9b7c6f2f45931bd29d16cf2616d00f",
        )
        model = darq_sq101()

    model_file = cache_clinicadl / FILE1.filename

    logger.info("Downloading quality check model.")

    if not (model_file.is_file()):
        try:
            model_file = fetch_file(FILE1, cache_clinicadl)
        except IOError as err:
            print("Unable to download required model for QC process:", err)

    # Load QC model
    logger.debug("Loading quality check model.")
    model.load_state_dict(torch.load(model_file))
    model.eval()
    if gpu:
        logger.debug("Working on GPU.")
        model = model.cuda()
    elif amp:
        raise ClinicaDLArgumentError(
            "AMP is designed to work with modern GPUs. Please add the --gpu flag."
        )

    with torch.no_grad():
        # Transform caps_dir in dict
        caps_dict = CapsDataset.create_caps_dict(caps_dir, multi_cohort=False)

        # Load DataFrame
        logger.debug("Loading data to check.")
        df = load_and_check_tsv(tsv_path, caps_dict, output_path.resolve().parent)

        dataset = QCDataset(caps_dir, df, use_tensor, use_uncropped_image)
        dataloader = DataLoader(
            dataset, num_workers=n_proc, batch_size=batch_size, pin_memory=True
        )

        columns = ["participant_id", "session_id", "pass_probability", "pass"]
        qc_df = pd.DataFrame(columns=columns)
        qc_df["pass"] = qc_df["pass"].astype(bool)
        softmax = torch.nn.Softmax(dim=1)
        logger.info(f"Quality check will be performed over {len(dataloader)} images.")

        for data in dataloader:
            logger.debug(f"Processing subject {data['participant_id']}.")
            inputs = data["image"]
            if gpu:
                inputs = inputs.cuda()
            with autocast(enabled=amp):
                outputs = softmax(model(inputs))
            # We cast back to 32bits. It should be a no-op as softmax is not eligible
            # to fp16 and autocast is forbidden on CPU (output would be bf16 otherwise).
            # But just in case...
            outputs = outputs.float()

            for idx, sub in enumerate(data["participant_id"]):
                pass_probability = outputs[idx, 1].item()
                row = [
                    [
                        sub,
                        data["session_id"][idx],
                        pass_probability,
                        pass_probability > threshold,
                    ]
                ]
                logger.debug(f"Quality score is {pass_probability}.")
                row_df = pd.DataFrame(row, columns=columns)
                qc_df = pd.concat([qc_df, row_df])

        qc_df.sort_values("pass_probability", ascending=False, inplace=True)
        qc_df.to_csv(output_path, sep="\t", index=False)
        logger.info(f"Results are stored at {output_path}.")
