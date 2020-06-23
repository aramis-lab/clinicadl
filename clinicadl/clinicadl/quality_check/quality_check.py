"""
This file contains all methods needed to perform the quality check procedure after t1-linear preprocessing.
"""
from os import path
import torch
import pandas as pd
from torch.utils.data import DataLoader

from clinicadl.quality_check.utils import QCDataset, resnet_qc_18


def quality_check(caps_dir, tsv_path, output_path, threshold=0.5, batch_size=1, num_workers=0, gpu=True,
                  use_extracted_tensors=False):
    if path.splitext(output_path)[1] != ".tsv":
        raise ValueError("Please provide an output path to a tsv file")

    # Load QC model
    script_dir = path.dirname(path.realpath(__file__))
    model = resnet_qc_18()
    model.load_state_dict(torch.load(path.join(script_dir, "model", "old_resnet18.pth.tar")))
    model.eval()
    if gpu:
        model.cuda()

    # Load DataFrame
    df = pd.read_csv(tsv_path, sep='\t')
    if ('session_id' not in list(df.columns.values)) or (
            'participant_id' not in list(df.columns.values)):
        raise Exception("the data file is not in the correct format."
                        "Columns should include ['participant_id', 'session_id']")
    dataset = QCDataset(caps_dir, df, use_extracted_tensors=use_extracted_tensors)
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

    qc_df.sort_values("pass_probability", axis=1)
    qc_df.to_csv(output_path, sep='\t', index=False)
