# coding: utf8

import pytest
from click.testing import CliRunner

from clinicadl.cmdline import cli


# Test to ensure that the help string, at the command line, is invoked without errors
# Test for the first level at the command line
@pytest.fixture(
    params=[
        "prepare-data",
        "generate",
        "interpret",
        "predict",
        "quality-check",
        "random-search",
        "train",
        "tsvtools",
    ]
)
def cli_args_first_lv(request):
    task = request.param
    return task


def test_first_lv(cli_args_first_lv):
    runner = CliRunner()
    task = cli_args_first_lv
    print(f"Testing input cli {task}")
    result = runner.invoke(cli, f"{task} -h")
    assert result.exit_code == 0


# Test for prepare-data cli, second level
@pytest.fixture(
    params=[
        "image",
        "slice",
        "patch",
        "roi",
    ]
)
def prepare_data_cli_arg1(request):
    return request.param


@pytest.fixture(
    params=[
        "t1-linear",
        "pet-linear",
        "custom",
    ]
)
def prepare_data_cli_arg2(request):
    return request.param


def test_second_lv_prepare_data(prepare_data_cli_arg1, prepare_data_cli_arg2):
    runner = CliRunner()
    arg1 = prepare_data_cli_arg1
    arg2 = prepare_data_cli_arg2
    print(f"Testing input prepare_data cli {arg1} {arg2}")
    result = runner.invoke(cli, f"prepare-data {arg1} {arg2} -h")
    assert result.exit_code == 0


# Test for the generate cli, second level
@pytest.fixture(
    params=[
        "shepplogan",
        "random",
        "trivial",
    ]
)
def generate_cli_arg1(request):
    return request.param


def test_second_lv_generate(generate_cli_arg1):
    runner = CliRunner()
    arg1 = generate_cli_arg1
    print(f"Testing input generate cli {arg1}")
    result = runner.invoke(cli, f"generate {arg1} -h")
    assert result.exit_code == 0


# Test for the interpret cli, second level
@pytest.fixture(
    params=[
        "",
    ]
)
def interpret_cli_arg1(request):
    return request.param


def test_second_lv_interpret(interpret_cli_arg1):
    runner = CliRunner()
    cli_input = interpret_cli_arg1
    print(f"Testing input generate cli {cli_input}")
    result = runner.invoke(cli, f"interpret {cli_input} -h")
    assert result.exit_code == 0


# Test for the predict cli, second level
@pytest.fixture(
    params=[
        "",
    ]
)
def predict_cli_arg1(request):
    return request.param


def test_second_lv_predict(predict_cli_arg1):
    runner = CliRunner()
    cli_input = predict_cli_arg1
    print(f"Testing input predict cli {cli_input}")
    result = runner.invoke(cli, f"predict {cli_input} -h")
    assert result.exit_code == 0


# Test for the train cli, second level
@pytest.fixture(
    params=[
        "classification",
        "regression",
        "reconstruction",
        "from_json",
        "resume",
        "list_models",
    ]
)
def train_cli_arg1(request):
    return request.param


def test_second_lv_train(train_cli_arg1):
    runner = CliRunner()
    cli_input = train_cli_arg1
    print(f"Testing input train cli {cli_input}")
    result = runner.invoke(cli, f"train {cli_input} -h")
    assert result.exit_code == 0


# Test for the random-search cli, second level
@pytest.fixture(params=["generate", "analysis"])
def rs_cli_arg1(request):
    task = request.param
    return task


def test_second_lv_random_search(rs_cli_arg1):
    runner = CliRunner()
    arg1 = rs_cli_arg1
    print(f"Testing input random-search cli {arg1}")
    result = runner.invoke(cli, f"random-search {arg1} -h")
    assert result.exit_code == 0


# Test for the quality-check cli, second level
@pytest.fixture(params=["t1-linear", "t1-volume"])
def qc_cli_arg1(request):
    task = request.param
    return task


def test_second_lv_quality_check(qc_cli_arg1):
    runner = CliRunner()
    arg1 = qc_cli_arg1
    print(f"Testing input quality-check cli {arg1}")
    result = runner.invoke(cli, f"quality-check {arg1} -h")
    assert result.exit_code == 0


# Test for the tsvtool cli, second level
@pytest.fixture(
    params=[
        "analysis",
        "get-labels",
        "kfold",
        "split",
        "prepare-experiment",
        "get-progression",
        "get-metadata",
    ]
)
def tsvtool_cli_arg1(request):
    return request.param


def test_second_lv_tsvtool(tsvtool_cli_arg1):
    runner = CliRunner()
    arg1 = tsvtool_cli_arg1
    print(f"Testing input tsvtools cli {arg1}")
    result = runner.invoke(cli, f"tsvtools {arg1} -h")
    assert result.exit_code == 0
