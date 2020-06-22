"""
This file contains all methods needed to perform the quality check procedure after t1-linear preprocessing.
"""
from os import path
import torch
import pandas as pd

from clinicadl.quality_check.utils import load_nifti_images, resnet_qc_18


def quality_check(caps_dir, tsv_path, output_path, threshold=0.5):

    if path.splitext(output_path)[1] != ".tsv":
        raise ValueError("Please provide an output path to a tsv file")

    # Load QC model
    script_dir = path.dirname(path.realpath(__file__))
    model = resnet_qc_18()
    model.load_state_dict(torch.load(path.join(script_dir, "model", "resnet18.pth")))
    model.eval()

    # Load DataFrame
    df = pd.read_csv(tsv_path, sep='\t')
    if ('session_id' not in list(df.columns.values)) or (
            'participant_id' not in list(df.columns.values)):
        raise Exception("the data file is not in the correct format."
                        "Columns should include ['participant_id', 'session_id']")

    columns = ['participant_id', 'session_id', 'pass_probability', 'pass']
    qc_df = pd.DataFrame(columns=columns)
    softmax = torch.nn.Softmax(dim=1)

    for _, row in df.iterrows():
        subject = row['participant_id']
        session = row['session_id']
        img_path = path.join(caps_dir, 'subjects', subject, session, 't1', 'preprocessing_dl',
                             '%s_%s_space-MNI_res-1x1x1.nii.gz'
                             % (subject, session))

        inputs = load_nifti_images(img_path)
        inputs = torch.cat(inputs).unsqueeze_(0)

        outputs = softmax.forward(model(inputs))
        pass_prob = outputs.data[0, 1]

        if pass_prob >= threshold:
            row = [[subject, session, pass_prob.item(), True]]
        else:
            row = [[subject, session, pass_prob.item(), False]]

        row_df = pd.DataFrame(row, columns=columns)
        qc_df = qc_df.append(row_df)

    qc_df.to_csv(output_path, sep='\t', index=False)


