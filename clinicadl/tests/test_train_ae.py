# coding: utf8

import pytest
import os
import shutil


@pytest.fixture(params=[
    'train_image_ae',
    'train_patch_ae',
    'train_roi_ae',
])
def cli_commands(request):
    if request.param == 'train_image_ae':
        test_input = [
            'train',
            'image',
            'autoencoder',
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
    elif request.param == 'train_patch_ae':
        test_input = [
            'train',
            'patch',
            'autoencoder',
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
    elif request.param == 'train_roi_ae':
        test_input = [
            'train',
            'roi',
            'autoencoder',
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
        os.path.join(
            "results",
            "best_model_dir",
            "fold_0",
            "ConvAutoencoder",
        ))
    assert performances_flag
    shutil.rmtree("results")
