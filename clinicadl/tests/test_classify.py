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

    if request.param == 'classify_image':
        test_input = [
            'classify',
            'data/classify/OASIS_test',
            'data/classify/OASIS_test/data.tsv',
            'data/models/image_model_baseline_AD_CN_single_fold/',
            '--prefix_output', 'DB-TEST',
            '-cpu'
        ]
    elif request.param == 'classify_slice':
        test_input = [
            'classify',
            'data/classify/OASIS_test',
            'data/classify/OASIS_test/data.tsv',
            'data/models/slice_model_baseline_AD_CN_single_fold/',
            '--prefix_output', 'DB-TEST',
            '-cpu'
        ]
    elif request.param == 'classify_patch':
        test_input = [
            'classify',
            'data/classify/OASIS_test',
            'data/classify/OASIS_test/data.tsv',
            'data/models/patch_model_baseline_AD_CN_multicnn_single_fold/',
            '--prefix_output', 'DB-TEST',
            '-cpu'
        ]
    else:
        raise NotImplementedError(
            "Test %s is not implemented." %
            request.param)

    return test_input


def test_classify(classify_commands):
    test_input = classify_commands

    flag_error = not os.system("clinicadl " + " ".join(test_input))
    output_file_img = exists(join('data/models/image_model_baseline_AD_CN_single_fold/fold-0/cnn_classification/best_balanced_accuracy/', 'DB-TEST_image_level_prediction.tsv'))
    #output_file_patch = exists(join('data/models/patch_model_baseline_AD_CN_multicnn_single_fold/fold-0/cnn_classification/best_balanced_accuracy/', 'DB-TEST_image_level_prediction.tsv'))
    #output_file_slice = exists(join('data/models/slice_model_baseline_AD_CN_single_fold/fold-0/cnn_classification/best_balanced_accuracy/', 'DB-TEST_image_level_prediction.tsv'))

    assert flag_error
    assert output_file_img
    #assert output_file_patch
    #assert output_file_slice
