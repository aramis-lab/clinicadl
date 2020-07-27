# coding: utf8

import pytest
import os
from os.path import join, exists


@pytest.fixture(params=[
    'classify_image',
    'classify_slice',
    'classify_patch'
])
def classify_commands(request):

    out_filename = 'fold-0/cnn_classification/best_balanced_accuracy/test-OASIS_image_level_prediction.tsv'
    if request.param == 'classify_image':
        data_folder = 'data/models/image_model_baseline_AD_CN_single_fold/'
        test_input = [
            'classify',
            'data/classify/OASIS_test',
            'data/classify/OASIS_test/data.tsv',
            data_folder,
            'test-OASIS',
            '-cpu'
        ]
        output_files = join(data_folder, out_filename)
    elif request.param == 'classify_slice':
        data_folder = 'data/models/slice_model_baseline_AD_CN_single_fold/'
        test_input = [
            'classify',
            'data/classify/OASIS_test',
            'data/classify/OASIS_test/data.tsv',
            data_folder,
            'test-OASIS',
            '-cpu'
        ]
        output_files = join(data_folder, out_filename)
    elif request.param == 'classify_patch':
        data_folder = 'data/models/patch_model_baseline_AD_CN_multicnn_single_fold/'
        test_input = [
            'classify',
            'data/classify/OASIS_test',
            'data/classify/OASIS_test/data.tsv',
            data_folder,
            'test-OASIS',
            '-cpu',
            '-nl'
        ]
        output_files = join(data_folder, out_filename)
    else:
        raise NotImplementedError(
            "Test %s is not implemented." %
            request.param)

    return test_input, output_files


def test_classify(classify_commands):
    test_input = classify_commands[0]
    output_files = classify_commands[1]

    flag_error = not os.system("clinicadl " + " ".join(test_input))

    assert flag_error
    assert exists(output_files)
