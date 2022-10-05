"""
This file contains all methods needed to perform the quality check procedure after t1-linear preprocessing.
"""
from logging import getLogger
from os import makedirs
from os.path import abspath, dirname, exists, join
from pathlib import Path

import pandas as pd
import torch
from clinica.utils.inputs import RemoteFileStructure, fetch_file
from torch.utils.data import DataLoader

from clinicadl.generate.generate_utils import load_and_check_tsv
from clinicadl.utils.caps_dataset.data import CapsDataset
from clinicadl.utils.exceptions import ClinicaDLArgumentError

from .utils import QCDataset, resnet_qc_18


def quality_check(
    caps_dir: str,
    output_path: str,
    tsv_path: str = None,
    threshold: float = 0.5,
    batch_size: int = 1,
    n_proc: int = 0,
    gpu: bool = True,
):

    logger = getLogger("clinicadl.quality_check")

    if not output_path.endswith(".tsv"):
        raise ClinicaDLArgumentError(f"Output path {output_path} must be a TSV file.")

    # Fetch QC model
    home = str(Path.home())
    cache_clinicadl = join(home, ".cache", "clinicadl", "models")
    url_aramis = "https://aramislab.paris.inria.fr/files/data/models/dl/qc/"
    logger.info("Downloading quality check model.")
    FILE1 = RemoteFileStructure(
        filename="resnet18.pth.tar",
        url=url_aramis,
        checksum="a97a781be3820b06424fe891ec405c78b87ad51a27b6b81614dbdb996ce60104",
    )

    makedirs(cache_clinicadl, exist_ok=True)

    model_file = join(cache_clinicadl, FILE1.filename)

    if not (exists(model_file)):
        try:
            model_file = fetch_file(FILE1, cache_clinicadl)
        except IOError as err:
            print("Unable to download required model for QC process:", err)

    # Load QC model
    logger.debug("Loading quality check model.")
    model = resnet_qc_18()
    model.load_state_dict(torch.load(model_file))
    model.eval()
    if gpu:
        logger.debug("Working on GPU.")
        model.cuda()

    # Transform caps_dir in dict
    caps_dict = CapsDataset.create_caps_dict(caps_dir, multi_cohort=False)

    # Load DataFrame
    logger.debug("Loading data to check.")
    df = load_and_check_tsv(tsv_path, caps_dict, dirname(abspath(output_path)))

    dataset = QCDataset(caps_dir, df)
    dataloader = DataLoader(
        dataset, num_workers=n_proc, batch_size=batch_size, pin_memory=True
    )

    columns = ["participant_id", "session_id", "pass_probability", "pass"]
    qc_df = pd.DataFrame(columns=columns)
    softmax = torch.nn.Softmax(dim=1)
    logger.info(f"Quality check will be performed over {len(dataloader)} images.")

    for data in dataloader:
        logger.debug(f"Processing subject {data['participant_id']}.")
        inputs = data["image"]
        if gpu:
            inputs = inputs.cuda()
        outputs = softmax.forward(model(inputs))

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
            qc_df = qc_df.append(row_df)

    qc_df.sort_values("pass_probability", ascending=False, inplace=True)
    qc_df.to_csv(output_path, sep="\t", index=False)
    logger.info(f"Results are stored at {output_path}.")
