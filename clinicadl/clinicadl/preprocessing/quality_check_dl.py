#! /usr/bin/env
# -*- coding: utf-8 -*-

import argparse
import sys
from os import path

from .quality_check_dl_utils import load_nifti_images
import torch
import pandas as pd

__author__ = "Junhao Wen"
__copyright__ = "Copyright 2018 The Aramis Lab Team"
__credits__ = ["Junhao Wen"]
__license__ = "See LICENSE.txt file"
__version__ = "0.1.0"
__maintainer__ = "Junhao Wen"
__email__ = "junhao.wen89@gmail.com"
__status__ = "Development"



def parse_options():
    parser = argparse.ArgumentParser(description='Apply automated QC',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("caps_dir", type=str,
                        help="the caps_directory for the outputs files of preprocessing pipeline")
    parser.add_argument("tsv", type=str,
                        help="the tsv file conatining participant_id and session_id")
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='The threshold of softmax layer to decide the label')

    params = parser.parse_args()

    return params


if __name__ == '__main__':
    params = parse_options()

    # Load QC model
    script_dir = path.dirname(sys.argv[0])
    model = torch.load(path.join(script_dir, "model", "resnet18.pth"))
    model.eval()

    # Load DataFrame
    df = pd.read_csv(params.tsv, sep='\t')
    if ('session_id' not in list(df.columns.values)) or ('participant_id' not in list(df.columns.values)):
        raise Exception("the data file is not in the correct format."
                        "Columns should include ['participant_id', 'session_id']")

    columns = ['participant_id', 'session_id', 'pass_probability', 'pass']
    qc_df = pd.DataFrame(columns=columns)
    softmax = torch.nn.Softmax(dim=1)

    for _, row in df.iterrows():
        subject = row['participant_id']
        session = row['session_id']
        img_path = path.join(params.caps_dir, 'subjects', subject, session, 't1_linear',
                             '%s_%s_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz'
                             % (subject, session))

        inputs = load_nifti_images(img_path)
        inputs = torch.cat(inputs).unsqueeze_(0)

        outputs = softmax.forward(model(inputs))
        pass_prob = outputs.data[0, 1]

        if pass_prob >= params.threshold:
            row = [[subject, session, pass_prob.item(), True]]
        else:
            row = [[subject, session, pass_prob.item(), False]]

        row_df = pd.DataFrame(row, columns=columns)
        qc_df = qc_df.append(row_df)

    qc_df.to_csv(params.output_path, sep='\t', index=False)
