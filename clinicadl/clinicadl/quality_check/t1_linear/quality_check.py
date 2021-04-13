"""
This file contains all methods needed to perform the quality check procedure after t1-linear preprocessing.
"""
from os import makedirs
from os.path import dirname, join, exists, splitext, abspath
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .utils import QCDataset, resnet_qc_18
from clinica.utils.inputs import fetch_file, RemoteFileStructure
from ...tools.data.utils import load_and_check_tsv
from ...tools.deep_learning.data import MRIDataset


def quality_check(caps_dir, output_path,
                  tsv_path=None, threshold=0.5,
                  batch_size=1, num_workers=0, gpu=True):

    if splitext(output_path)[1] != ".tsv":
        raise ValueError("Please provide an output path to a tsv file")

    # Fetch QC model
    home = str(Path.home())
    cache_clinicadl = join(home, '.cache', 'clinicadl', 'models')
    url_aramis = 'https://aramislab.paris.inria.fr/files/data/models/dl/qc/'
    FILE1 = RemoteFileStructure(
        filename='resnet18.pth.tar',
        url=url_aramis,
        checksum='a97a781be3820b06424fe891ec405c78b87ad51a27b6b81614dbdb996ce60104'
    )

    makedirs(cache_clinicadl, exist_ok=True)

    model_file = join(cache_clinicadl, FILE1.filename)

    if not(exists(model_file)):
        try:
            model_file = fetch_file(FILE1, cache_clinicadl)
        except IOError as err:
            print('Unable to download required model for QC process:', err)

    # Load QC model
    model = resnet_qc_18()
    model.load_state_dict(torch.load(model_file))
    model.eval()
    if gpu:
        model.cuda()

    # Transform caps_dir in dict
    caps_dict = MRIDataset.create_caps_dict(caps_dir, multi_cohort=False)

    # Load DataFrame
    df = load_and_check_tsv(tsv_path, caps_dict, dirname(abspath(output_path)))

    dataset = QCDataset(caps_dir, df)
    dataloader = DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        pin_memory=True
    )

    columns = ['participant_id', 'session_id', 'pass_probability', 'pass']
    qc_df = pd.DataFrame(columns=columns)
    softmax = torch.nn.Softmax(dim=1)

    for data in dataloader:
        inputs = data['image']
        if gpu:
            inputs = inputs.cuda()
        outputs = softmax.forward(model(inputs))

        for idx, sub in enumerate(data['participant_id']):
            pass_probability = outputs[idx, 1].item()
            row = [[sub, data['session_id'][idx], pass_probability, pass_probability > threshold]]
            row_df = pd.DataFrame(row, columns=columns)
            qc_df = qc_df.append(row_df)

    qc_df.sort_values("pass_probability", ascending=False, inplace=True)
    qc_df.to_csv(output_path, sep='\t', index=False)
