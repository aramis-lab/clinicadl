# coding: utf8

import os
import shutil
from os import path

import nibabel as nib
import numpy as np
import pytest

caps_dir = "data/dataset/random_example"


@pytest.fixture(params=["train_1roi", "train_2roi"])
def cli_commands(request):

    if request.param == "train_1roi":
        roi_list = ["random1"]
        test_input = [
            "train",
            "classification",
            caps_dir,
            "data/labels_list",
            "results",
            "--epochs",
            "1",
            "--n_splits",
            "2",
            "--split",
            "0",
            "--roi_list",
            "random1",
        ]
    elif request.param == "train_2roi":
        roi_list = ["random1", "random2"]
        test_input = [
            "train",
            "classification",
            caps_dir,
            "data/labels_list",
            "results",
            "--epochs",
            "1",
            "--n_splits",
            "2",
            "--split",
            "0",
            "--roi_list",
            "random1 random2",
            "--multi",
        ]
    else:
        raise NotImplementedError("Test %s is not implemented." % request.param)

    return roi_list, test_input


def test_train(cli_commands):
    if os.path.exists("results"):
        shutil.rmtree("results")

    roi_list, test_input = cli_commands
    crop_size = (50, 50, 50)
    os.makedirs(
        path.join(caps_dir, "masks", "roi_based", "tpl-MNI152NLin2009cSym"),
        exist_ok=True,
    )
    for roi in roi_list:
        filename = "tpl-MNI152NLin2009cSym_desc-Crop_res-1x1x1_roi-%s_mask.nii.gz" % roi
        crop_center = np.random.randint(low=55, high=100, size=3)
        mask_np = np.zeros((1, 169, 208, 179))
        mask_np[
            :,
            crop_center[0] - crop_size[0] // 2 : crop_center[0] + crop_size[0] // 2 :,
            crop_center[1] - crop_size[1] // 2 : crop_center[1] + crop_size[1] // 2 :,
            crop_center[2] - crop_size[2] // 2 : crop_center[2] + crop_size[2] // 2 :,
        ] = 1
        mask_nii = nib.Nifti1Image(mask_np, affine=np.eye(4))
        nib.save(
            mask_nii,
            path.join(
                caps_dir, "masks", "roi_based", "tpl-MNI152NLin2009cSym", filename
            ),
        )

    flag_error = not os.system("clinicadl " + " ".join(test_input))
    performances_flag = os.path.exists(
        os.path.join("results", "fold-0", "best-loss", "train")
    )
    assert flag_error
    assert performances_flag
    shutil.rmtree("results")
    shutil.rmtree(path.join(caps_dir, "masks"))
