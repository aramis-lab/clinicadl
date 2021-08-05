# coding: utf8

import pytest

import clinicadl.cmdline as cli


@pytest.fixture(
    params=[
        "extract",
        "generate",
        "interpret",
        "predict",
        "random_search"
        "quality_check",
        "train",
        "tsvtool",
    ]
)
def generate_cli_commands(request):

    # fmt: off
    if request.param == 'extract':
        test_input = [
            'extract',
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
            '/dir/model_path/',
            'DB_XXXXX',
            '--caps_directory', '/dir/caps',
            '--tsv_path', '/dir/tsv_file',
        ]
        keys_output = [
            'task',
            'model_path',
            'data_group',
            'caps_directory',
            'tsv_path',
        ]

    if request.param == 'interpret':
        test_input = [
            'interpret',
            '/dir/model_path/',
            'DB_XXXXX',
            'GradCAM',
            '--caps_directory', '/dir/caps',
            '--tsv_path', '/dir/tsv_file',
        ]
        keys_output = [
            'task',
            'model_path',
            'data_group',
            'name',
            'caps_directory',
            'tsv_path',
        ]

    if request.param == 'train':
        test_input = [
            'train',
            'classification',
            '/dir/caps',
            '/dir/caps/preprocessing.json',
            '/dir/tsv_path/',
            '/dir/output/']
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

    test_input, keys_output = generate_cli_commands
    print("Value of test_input is:", type(test_input), test_input)
    regex = re.compile(r"\-.*$")
    test_input_filtered = [i for i in test_input if not regex.match(i)]
    parser = cli.parse_command_line()
    args = parser.parse_args(test_input)
    arguments = vars(args)
    outputs = [str(arguments[x]) for x in keys_output]
    print(outputs)
    assert outputs == test_input_filtered
