# coding: utf8

import pytest
from click.testing import CliRunner

import clinicadl.cmdline as cli

# Test to ensure that the help string at the command line is invoked without errors

# Test for the first level at the command line
@pytest.fixture(
    params=[
        "extract",
        "generate",
        "interpret",
        "predict",
        # "random-search"
        "quality-check",
        "train",
        "tsvtool",
    ]
)
def generate_cli_first_lv(request):
    task = request.param
    return task


def test_first_lv(generate_cli_first_lv):
    runner = CliRunner()
    cli_input = generate_cli_first_lv
    print(f"Testing input cli {cli_input}")
    result = runner.invoke(cli, f"{cli_input} -h")
    assert result.exit_code == 0


# Test for the generate cli (second level)
@pytest.fixture(
    params=[
        "shepplogan",
        "random",
        "trivial",
    ]
)
def generate_cli_second_lv_generate(request):
    task = request.param
    return task


def test_second_lv_generate(generate_cli_second_lv_generate):
    runner = CliRunner()
    cli_input = generate_cli_second_lv_generate
    print(f"Testing input cli generate {cli_input}")
    result = runner.invoke(cli, f"generate {cli_input} -h")
    assert result.exit_code == 0


@pytest.fixture(params=["t1-linear", "t1-volume"])
def generate_cli_second_lv_quality_check(request):
    task = request.param
    return task


def test_second_lv_quality_check(generate_cli_second_lv_quality_check):
    runner = CliRunner()
    cli_input = generate_cli_second_lv_quality_check
    print(f"Testing input cli quality-check {cli_input}")
    result = runner.invoke(cli, f"quality-check {cli_input} -h")
    assert result.exit_code == 0


@pytest.fixture(
    params=[
        "analysis",
        "getlabels",
        "kfold",
        "restrict",
        "split",
    ]
)
def generate_cli_second_lv_tsvtool(request):
    task = request.param
    return task


def test_second_lv_tsvtool(generate_cli_second_lv_tsvtool):
    runner = CliRunner()
    cli_input = generate_cli_second_lv_tsvtool
    print(f"Testing input cli tsvtool {cli_input}")
    result = runner.invoke(cli, f"tsvtool {cli_input} -h")
    assert result.exit_code == 0
