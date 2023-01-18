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

from .sq101 import squeezenet_qc as darq_sq101
from .utils import QCDataset
from .utils import resnet_qc_18 as deep_r18
from .utils_bis import resnet_qc_18 as darq_r18


def quality_check(
    caps_dir: str,
    output_path: str,
    tsv_path: str = None,
    threshold: float = 0.5,
    batch_size: int = 1,
    n_proc: int = 0,
    gpu: bool = True,
    network: str = "darq",
    use_tensor: bool = True,
):

    logger = getLogger("clinicadl.quality_check")

    if not output_path.endswith(".tsv"):
        raise ClinicaDLArgumentError(f"Output path {output_path} must be a TSV file.")

    # Fetch QC model
    home = str(Path.home())

    if network == "deep_qc":
        cache_clinicadl = join(home, ".cache", "clinicadl", "models")
        url_aramis = "https://aramislab.paris.inria.fr/files/data/models/dl/qc/"
        FILE1 = RemoteFileStructure(
            filename="resnet18.pth.tar",
            url=url_aramis,
            checksum="a97a781be3820b06424fe891ec405c78b87ad51a27b6b81614dbdb996ce60104",
        )
        makedirs(cache_clinicadl, exist_ok=True)
        model_file = join(cache_clinicadl, FILE1.filename)
        model = deep_r18()

    if network == "darq":
        model_file = "/Users/camille.brianceau/Desktop/QC/code/models/python_DARQ/cls/model_r18/best_tnr.pth"
        model = darq_r18()

    if network == "sq101":
        model_file = "/network/lustre/iss02/aramis/users/camille.brianceau/QC/models2/Deep_QC/model_sq101/best_tnr_cpu.pth"
        model = darq_sq101()

    url_r18_2018 = "/Users/camille.brianceau/Desktop/QC/code/models/Deep-QC/model_r18/best_tnr_cpu.pth"
    url_r152_2022 = (
        "/Users/camille.brianceau/Desktop/QC/code/models/DARQ/model_r152/best_tnr.pth"
    )
    logger.info("Downloading quality check model.")

    if not (exists(model_file)):
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

    with torch.no_grad():
        # Transform caps_dir in dict
        caps_dict = CapsDataset.create_caps_dict(caps_dir, multi_cohort=False)

        # Load DataFrame
        logger.debug("Loading data to check.")
        df = load_and_check_tsv(tsv_path, caps_dict, dirname(abspath(output_path)))

        dataset = QCDataset(caps_dir, df, use_tensor)
        print(dataset)
        dataloader = DataLoader(
            dataset, num_workers=n_proc, batch_size=batch_size, pin_memory=True
        )

        columns = ["participant_id", "session_id", "pass_probability", "pass"]
        qc_df = pd.DataFrame(columns=columns)
        softmax = torch.nn.Softmax(dim=1)
        logger.info(f"Quality check will be performed over {len(dataloader)} images.")

        for data in dataloader:
            # print(data)
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
