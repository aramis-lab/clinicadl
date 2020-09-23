"""
This file contains all methods needed to perform the quality check procedure after t1-linear preprocessing.
"""
from os import pardir, makedirs
from os.path import dirname, join, abspath, split, exists, splitext
from pathlib import Path
import torch
import pandas as pd
from torch.utils.data import DataLoader
from clinica.iotools.utils.data_handling import create_subs_sess_list

from clinicadl.quality_check.utils import QCDataset, resnet_qc_18
from clinicadl.tools.inputs.input import fetch_file
from clinicadl.tools.inputs.input import RemoteFileStructure
from clinicadl.tools.data.utils import load_and_check_tsv


def quality_check(caps_dir, output_path, preprocessing,
                  tsv_path=None, threshold=0.5, batch_size=1, num_workers=0, gpu=True):
    if preprocessing != "t1-linear":
        raise NotImplementedError("The quality check procedure implemented in clinicadl is meant to be run "
                                  "on t1-linear preprocessing only.")

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

    if not(exists(cache_clinicadl)):
        makedirs(cache_clinicadl)

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

    # Load DataFrame
    if tsv_path is not None:
        df = pd.read_csv(tsv_path, sep='\t')
        if ('session_id' not in list(df.columns.values)) or (
                'participant_id' not in list(df.columns.values)):
            raise Exception("the data file is not in the correct format."
                            "Columns should include ['participant_id', 'session_id']")
    else:
        create_subs_sess_list(caps_dir, output_path, is_bids_dir=False, use_session_tsv=False)
        df = pd.read_csv(join(output_path, 'subjects_sessions_list.tsv'), sep="\t")

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
