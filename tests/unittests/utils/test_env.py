import os
import re
from unittest.mock import MagicMock, call, patch

import pytest

from clinicadl.utils.env import (
    _conda_prefix,
    _export_conda_environment,
    _freeze_packages,
    _get_environment_info,
    dump_environment,
)

PIP_FREEZE = "numpy==2.1.0\ntorch==2.3.0\n"
CONDA_YML = (
    "name: clinicadl_dev\nchannels:\n  - conda-forge\n"
    "dependencies:\n  - python=3.12.0=h123\n  - pip:\n      - numpy==2.1.0\n"
)
CONDA_PORTABLE_YML = (
    "name: clinicadl_dev\nchannels:\n  - conda-forge\n"
    "dependencies:\n  - python=3.12.0\n  - pip:\n      - numpy==2.1.0\n"
)


def _completed(stdout):
    """A stand-in for the object returned by ``subprocess.run``."""
    return MagicMock(stdout=stdout, stderr="", returncode=0)


def test_get_environment_info():
    info = _get_environment_info()
    expected_keys = {
        "timestamp",
        "python_version",
        "python_implementation",
        "executable",
        "platform",
        "machine",
        "processor",
    }
    assert expected_keys == set(info)
    assert all(isinstance(value, str) for value in info.values())


@patch.dict(os.environ, {}, clear=True)
def test_conda_prefix_absent():
    assert _conda_prefix() is None


@patch("clinicadl.utils.env.sys")
@patch.dict(os.environ, {"CONDA_PREFIX": "/opt/conda/envs/x"}, clear=True)
def test_conda_prefix_matches_interpreter(mock_sys):
    mock_sys.prefix = "/opt/conda/envs/x"
    assert _conda_prefix() == "/opt/conda/envs/x"


@patch.dict(
    os.environ, {"CONDA_PREFIX": "/definitely/not/the/interpreter/prefix"}, clear=True
)
def test_conda_prefix_mismatch(caplog):
    # CONDA_PREFIX is set but does not match the running interpreter
    with caplog.at_level("DEBUG"):
        assert _conda_prefix() is None
    assert re.match(
        r"CONDA_PREFIX \(/definitely/not/the/interpreter/prefix\) does not match the running interpreter \(.*\); falling back to a pip snapshot.",
        caplog.records[0].message,
    )


@patch("clinicadl.utils.env.subprocess.run")
def test_freeze_packages(mock_run):
    mock_run.return_value = _completed(PIP_FREEZE)
    assert _freeze_packages() == PIP_FREEZE
    command = mock_run.call_args.args[0]
    assert command[1:] == ["-m", "pip", "freeze"]


@patch("clinicadl.utils.env.subprocess.run", side_effect=FileNotFoundError)
def test_freeze_packages_no_pip(mock_run):
    with pytest.raises(RuntimeError, match="Could not run 'pip'"):
        _freeze_packages()


@patch.dict(os.environ, {"CONDA_EXE": "/opt/conda/bin/conda"}, clear=True)
@patch("clinicadl.utils.env.subprocess.run")
def test_export_conda_environment_accurate(mock_run):
    mock_run.return_value = _completed(CONDA_YML)
    assert _export_conda_environment("/opt/conda/envs/x") == CONDA_YML
    command = mock_run.call_args.args[0]
    assert command == [
        "/opt/conda/bin/conda",
        "env",
        "export",
        "-p",
        "/opt/conda/envs/x",
    ]


@patch.dict(os.environ, {"CONDA_EXE": "/opt/conda/bin/conda"}, clear=True)
@patch("clinicadl.utils.env.subprocess.run")
def test_export_conda_environment_portable(mock_run):
    mock_run.return_value = _completed(CONDA_YML)
    _export_conda_environment("/opt/conda/envs/x", portable=True)
    command = mock_run.call_args.args[0]
    assert "--no-builds" in command
    assert "--ignore-channels" in command


@patch("clinicadl.utils.env._conda_prefix", return_value=None)
@patch("clinicadl.utils.env._freeze_packages", return_value=PIP_FREEZE)
def test_dump_environment_pip(mock_freeze, mock_prefix, tmp_path):
    dump_environment("environment", tmp_path)
    path = tmp_path / "environment.txt"
    content = path.read_text()
    assert PIP_FREEZE in content
    # the header is made of comment lines, valid in a requirements file
    header = [line for line in content.splitlines() if line.startswith("#")]
    assert any("python_version" in line for line in header)
    assert any(f"pip install -r {path}" in line for line in header)


def _fake_export(prefix, portable=False):
    """Return the portable or accurate export depending on ``portable``, not call order."""
    return CONDA_PORTABLE_YML if portable else CONDA_YML


@patch("clinicadl.utils.env._conda_prefix", return_value="/opt/conda/envs/x")
@patch("clinicadl.utils.env._freeze_packages", return_value=PIP_FREEZE)
@patch("clinicadl.utils.env._export_conda_environment", side_effect=_fake_export)
def test_dump_environment_conda(mock_export, mock_freeze, mock_prefix, tmp_path):
    dump_environment("env", tmp_path)
    assert (tmp_path / "env.txt").is_file()
    path = tmp_path / "env.yml"
    content = path.read_text()
    assert CONDA_YML in content
    path = tmp_path / "env_portable.yml"
    content = path.read_text()
    assert CONDA_PORTABLE_YML in content

    assert mock_export.call_args_list == [
        call("/opt/conda/envs/x", portable=False),
        call("/opt/conda/envs/x", portable=True),
    ]
    header = [line for line in content.splitlines() if line.startswith("#")]
    assert any(f"conda env create -f {path}" in line for line in header)


@patch("clinicadl.utils.env._conda_prefix", return_value=None)
@patch("clinicadl.utils.env._freeze_packages", return_value=PIP_FREEZE)
def test_dump_environment_creates_parent_dirs(mock_freeze, mock_prefix, tmp_path):
    dump_environment("environment", tmp_path / "nested" / "dir")
    assert (tmp_path / "nested" / "dir").exists()
