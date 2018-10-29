#! /usr/bin/env
# -*- coding: utf-8 -*-

import argparse
import sys

from quality_check_dl_utils import load_nifti_images
from model.util import *
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.autograd import Variable

default_data_dir = os.path.dirname(sys.argv[0])


def parse_options():
    parser = argparse.ArgumentParser(description='Apply automated QC',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--ref", action="store_true", default=False,
                        help="Use reference images")
    parser.add_argument("--caps_dir", type=str,
                        help="the caps_directory for the outputs files of preprocessing pipeline")
    parser.add_argument("--tsv", type=str,
                        help="the tsv file conatining participant_id and session_id")
    parser.add_argument("--load", type=str, default=default_data_dir + os.sep + 'python/model_r18/best_tnr_cpu.pth',
                        help="Load pretrained model (mondatory)")
    parser.add_argument("--net", choices=['r18', 'r34', 'r50', 'r101', 'r152', 'sq101'],
                        help="Network type", default='r18')
    parser.add_argument('--raw', action="store_true", default=False,
                        help='Print raw score [0:1]')
    parser.add_argument('-q', '--quiet', action="store_true", default=False,
                        help='Quiet mode, set status code to 0 - Pass, 1 - fail')

    params = parser.parse_args()

    return params


if __name__ == '__main__':
    params = parse_options()
    use_ref = False

    if params.load is None:
        print("need to provide pre-trained model!")
        exit(1)

    df = pd.read_csv(params.tsv, sep='\t')
    if ('diagnosis' != list(df.columns.values)[2]) and ('session_id' != list(df.columns.values)[1]) and (
                'participant_id' != list(df.columns.values)[0]):
        raise Exception('the data file is not in the correct format.')
    img_list = list(df['participant_id'])
    sess_list = list(df['session_id'])

    ### creat a new df to store the QC results.
    qc_df = pd.DataFrame(columns=df.columns)

    for i in range(len(img_list)):
        img_path = os.path.join(params.caps_dir, 'subjects', img_list[i], sess_list[i], 't1', 'preprocessing_dl',
                                img_list[i] + '_' + sess_list[i] + '_space-MNI_res-1x1x1_linear_registration.nii.gz')

        print("Quality check for subject: %s_%s" % (img_list[i], sess_list[i]))
        inputs = load_nifti_images(img_path)

        model = get_qc_model(params, use_ref=use_ref)
        model.train(False)

        # convert inputs into properly formated tensor
        # with a single batch dimension
        inputs = torch.cat(inputs).unsqueeze_(0)
        inputs = Variable(inputs)

        softmax = nn.Softmax(dim=1)
        outputs = softmax.forward(model(inputs))
        _, preds = torch.max(outputs.data, 1)

        # raw score
        if params.raw:
            print(outputs.data[0, 1])
        elif not params.quiet:
            if preds.item() == 1:
                print("Pass!! QC Results for subject: %s_%s" % (img_list[i], sess_list[i]))
            else:
                print("Fail!!! QC Results for subject: %s_%s" % (img_list[i], sess_list[i]))
        else:
            exit(0 if preds.item() == 1 else 1)

        row = list([img_list[i], sess_list[i], str(preds.item())])
        row = np.array(row).reshape(1, len(row))
        row_df = pd.DataFrame(row, columns=df.columns)
        qc_df = qc_df.append(row_df)

    qc_df.reset_index(inplace=True, drop=True)
    qc_df.to_csv(os.path.join(params.caps_dir, 'qc_dl.tsv'), sep='\t')