# coding: utf8

import pytest

import clinicadl.cli as cli


@pytest.fixture(
    params=[
        "preprocessing_run_t1_linear",
        "preprocessing_run_t1_extensive",
        "extract_tensor",
        "generate",
        "quality_check",
        "predict",
        "train_image",
        "train_slice",
        "train_patch",
    ]
)
def generate_cli_commands(request):
    if request.param == "preprocessing_run_t1_linear":
        test_input = ["preprocessing", "run", "t1-linear", "/dir/bids/", "/dir/caps/"]
        keys_output = [
            "task",
            "preprocessing_task",
            "preprocessing",
            "bids_directory",
            "caps_directory",
        ]

    if request.param == "preprocessing_run_t1_extensive":
        test_input = ["preprocessing", "run", "t1-extensive", "/dir/caps/"]
        keys_output = ["task", "preprocessing_task", "preprocessing", "caps_directory"]

    # fmt: off
    if request.param == 'extract_tensor':
        test_input = [
            'preprocessing',
            'extract-tensor',
            '/dir/caps',
            't1-linear',
            'slice',
            '-ps', '50',
            '-ss', '50',
            '-sd', '0',
            '-sm', 'rgb',
            '-tsv', '/dir/tsv.file',
            '-wd', '/dir/work/dir',
            '-np', '1',
        ]
        keys_output = [
            'task',
            'preprocessing_task',
            'caps_directory',
            'modality',
            'extract_method',
            'patch_size',
            'stride_size',
            'slice_direction',
            'slice_mode',
            'subjects_sessions_tsv',
            'working_directory',
            'n_procs',
        ]
    if request.param == 'quality_check':
        test_input = [
            'preprocessing',
            'quality-check',
            't1-linear',
            '/dir/caps',
            '/dir/res.tsv',
            '-tsv', '/dir/tsv.file',
            '--threshold', '0.5',
            '--batch_size', '8']
        keys_output = [
            'task',
            'preprocessing_task',
            'preprocessing',
            'caps_directory',
            'output_path',
            'subjects_sessions_tsv',
            'threshold',
            'batch_size']

    if request.param == 'generate':
        test_input = [
            'generate',
            'random',
            '/dir/caps',
            't1-linear',
            '/dir/output/',
            '--n_subjects', '10',
            '--mean', '0.5',
            '--sigma', '0.5']
        keys_output = [
            'task',
            'mode',
            'caps_directory',
            'preprocessing',
            'output_dir',
            'n_subjects',
            'mean',
            'sigma']

    if request.param == 'predict':
        test_input = [
            'predict',
            '/dir/caps',
            '/dir/tsv_file',
            '/dir/model_path/',
            'DB_XXXXX'
        ]
        keys_output = [
            'task',
            'caps_directory',
            'tsv_path',
            'model_path',
            'prefix_output'
        ]

    if request.param == 'train_slice':
        test_input = [
            'train',
            'slice',
            'classification',
            '/dir/caps',
            't1-linear',
            '/dir/tsv_path/',
            '/dir/output/',
            'Conv5_FC3']
        keys_output = [
            'task',
            'mode',
            'network_task',
            'caps_directory',
            'preprocessing',
            'tsv_path',
            'output_dir',
            'model']

    if request.param == 'train_image':
        test_input = [
            'train',
            'image',
            'reconstruction',
            '/dir/caps',
            't1-linear',
            '/dir/tsv_path/',
            '/dir/output/',
            'Conv5_FC3']
        keys_output = [
            'task',
            'mode',
            'network_task',
            'caps_directory',
            'preprocessing',
            'tsv_path',
            'output_dir',
            'model']
    if request.param == 'train_patch':
        test_input = [
            'train',
            'patch',
            'regression',
            '/dir/caps',
            't1-linear',
            '/dir/tsv_path/',
            '/dir/output/',
            'Conv5_FC3']
        keys_output = [
            'task',
            'mode',
            'network_task',
            'caps_directory',
            'preprocessing',
            'tsv_path',
            'output_dir',
            'model']
    # fmt: on

    return test_input, keys_output


def test_cli(generate_cli_commands):
    import re

    test_input = generate_cli_commands[0]
    keys_output = generate_cli_commands[1]
    print("Value of test_input is:", type(test_input), test_input)
    regex = re.compile(r"\-.*$")
    test_input_filtered = [i for i in test_input if not regex.match(i)]
    parser = cli.parse_command_line()
    args = parser.parse_args(test_input)
    arguments = vars(args)
    outputs = [str(arguments[x]) for x in keys_output]
    print(outputs)
    assert outputs == test_input_filtered
