# coding: utf8

import pytest
import os
import shutil


@pytest.fixture(params=[
    'classify_image',
    'classify_slice',
    'classify_patch'
])
def cli_commands(request):

    if request.param == 'classify_image':
        test_input = [
            'classify',    
            'data/OASIS_test',
            'data/OASIS_test/data.tsv',
            'data/models/3D_subject/baseline/AD_CN/fold_0/',
            'data/output/',
            '-cpu'
        ]
    elif request.param == 'classify_slice':
        test_input = [
            'classify',    
            'data/OASIS_test',
            'data/OASIS_test/data.tsv',
            'data/models/2D_slice/baseline/AD_CN/best_model/fold_0/',
            'data/output/',
            '-cpu'
        ]
    elif request.param == 'classify_patch':
        test_input = [
            'classify',    
            'data/OASIS_test',
            'data/OASIS_test/data.tsv',
            'data/models/3D_patch/baseline/AD_CN/best_model/fold_0/',
            'data/output/',
            '-cpu'
        ]
    else:
        raise NotImplementedError(
            "Test %s is not implemented." %
            request.param)

    return test_input


def test_train(cli_commands):
    test_input = cli_commands
    
    flag_error = not os.system("clinicadl " + " ".join(test_input))
    
    assert flag_error
