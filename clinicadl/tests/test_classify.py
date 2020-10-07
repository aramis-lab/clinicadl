# coding: utf8

import pytest
import os
from os.path import join, exists


@pytest.fixture(params=[
    'classify_image',
    'classify_roi',
    'classify_slice',
    'classify_patch'])

def classify_commands(request):
    out_filename = 'fold-0/cnn_classification/best_balanced_accuracy/test-RANDOM_image_level_prediction.tsv'
    if request.param == 'classify_image':
        model_folder = 'data/models/model_exp3_splits_1/'
        test_input = [
            'classify',
            'data/dataset/random_example',
            'data/dataset/random_example/data.tsv',
            model_folder,
            'test-RANDOM',
            '-cpu'
        ]
        output_files = join(model_folder, out_filename)
    elif request.param == 'classify_slice':
        model_folder = 'data/models/model_exp18_splits_1/'
        test_input = [
            'classify',
            'data/dataset/random_example',
            'data/dataset/random_example/data.tsv',
            model_folder,
            'test-RANDOM',
            '-cpu'
        ]
        output_files = join(model_folder, out_filename)
    elif request.param == 'classify_patch':
        model_folder = 'data/models/model_exp14_splits_1/'
        test_input = [
            'classify',
            'data/dataset/random_example',
            'data/dataset/random_example/data.tsv',
            model_folder,
            'test-RANDOM',
            '-cpu',
            '-nl'
        ]
        output_files = join(model_folder, out_filename)
    elif request.param == 'classify_roi':
        model_folder = 'data/models/model_exp8_splits_1/'
        test_input = [
            'classify',
            'data/dataset/random_example',
            'data/dataset/random_example/data.tsv',
            model_folder,
            'test-RANDOM',
            '-cpu',
            '-nl'
        ]
        output_files = join(model_folder, out_filename)
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
