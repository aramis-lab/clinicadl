# coding: utf8

import pytest
import os
import shutil


@pytest.fixture(params=[
    'train_slice',
    'train_image_cnn',
    'train_patch_cnn',
    'train_patch_multicnn',
    'train_roi_cnn',
])
def cli_commands(request):

    if request.param == 'train_slice':
        test_input = [
            'train',
            'slice',
            'data/OASIS_test',
            't1-linear'
            'data/labels_list',
            'results',
            'resnet18',
            '--epochs', '1',
            '--n_splits', '2',
            '--split', '0',
            '-cpu'
        ]
    elif request.param == 'train_image_cnn':
        test_input = [
            'train',
            'image',
            'cnn',
            'data/OASIS_test',
            't1-linear',
            'data/labels_list',
            'results',
            'Conv5_FC3',
            '--epochs', '1',
            '--n_splits', '2',
            '--split', '0',
            '-cpu'
        ]
    elif request.param == 'train_patch_cnn':
        test_input = [
            'train',
            'patch',
            'cnn',
            'data/OASIS_test',
            't1-linear'
            'data/labels_list',
            'results',
            'Conv4_FC3',
            '--epochs', '1',
            '--n_splits', '2',
            '--split', '0',
            '-cpu'
        ]
    elif request.param == 'train_patch_multicnn':
        test_input = [
            'train',
            'patch',
            'multicnn',
            'data/OASIS_test',
            't1-linear',
            'data/labels_list',
            'results',
            'Conv4_FC3',
            '--epochs', '1',
            '--n_splits', '2',
            '--split', '0',
            '-cpu'
        ]
    elif request.param == 'train_roi_cnn':
        test_input = [
            'train',
            'roi',
            'cnn',
            'data/OASIS_test',
            't1-linear',
            'data/labels_list',
            'results',
            'Conv4_FC3',
            '--epochs', '1',
            '--n_splits', '2',
            '--split', '0',
            '-cpu'
        ]
    else:
        raise NotImplementedError(
            "Test %s is not implemented." %
            request.param)

    return test_input


def test_train(cli_commands):
    test_input = cli_commands
    os.system("clinicadl " + " ".join(test_input))
    performances_flag = os.path.exists(
        os.path.join("results", "performances", "fold_0"))
    assert performances_flag
    shutil.rmtree("results")
