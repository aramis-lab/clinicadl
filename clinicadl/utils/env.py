import logging
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from clinicadl.utils.dictionary.words import PORTABLE

logger = logging.getLogger(__name__)

ENCODING = "utf-8"


def dump_environment(
    filename_prefix: str,
    root: Path,
) -> None:
    """
    Write files describing the current environment, for later reproduction.

    A ``pip freeze`` snapshot is always produced in a ``.txt`` file. If inside a conda environment
    (``CONDA_PREFIX`` is set), two ``.yml`` files are produced via ``conda env export``:

    - an accurate file with the exact pins for reproducing the
      environment on the *same machine*. Build strings and
      channels are kept (``conda env export``), so the solver resolves the
      very same artifacts. Best when the same user reruns on the same
      machine.
    - a portable file with looser pins that are very likely to
      solve on another machine. Build strings and channel pins are
      dropped (``conda env export --no-builds --ignore-channels``) while
      exact versions are kept.

    .. note::
    ``pip freeze`` already pins exact versions and resolves per-platform wheels.

    In all files, a commented header records the Python version, platform and timestamp.

    Parameters
    ----------
    filename_prefix : str
        The prefix of the names of the destination files. The output files are saved in
        ``<filename_prefix>.txt``, ``<filename_prefix>.yml`` and ``<filename_prefix>_portable.yml``
    root : Path
        The directory where the files will be saved.

    Notes
    -----
    No artifact reproduces the interpreter version or system-level libraries
    (CUDA, BLAS, ...). The header records the Python version and platform so that a
    divergence is visible, but matching them is left to the user. Even the
    portable conda export keeps OS-specific low-level packages, so a macOS
    export may still need manual trimming to solve on Linux.
    """
    info = _get_environment_info()

    prefix = _conda_prefix()
    if prefix is not None:
        path = (root / filename_prefix).with_suffix(".yml")
        recreate = f"conda env create -f {path}"
        body = _export_conda_environment(prefix, portable=False)
        _write_environment(info, path, body, recreate)

        path = (root / "_".join([filename_prefix, PORTABLE])).with_suffix(".yml")
        recreate = f"conda env create -f {path}"
        body = _export_conda_environment(prefix, portable=True)
        _write_environment(info, path, body, recreate)

    path = (root / filename_prefix).with_suffix(".txt")
    body = _freeze_packages()
    recreate = f"pip install -r {path}"
    _write_environment(info, path, body, recreate)


def _write_environment(
    info: dict[str, str], path: Path, body: str, recreate: str
) -> None:
    header_lines = ["# ClinicaDL environment snapshot"]
    header_lines += [f"# {key}: {value}" for key, value in info.items()]
    header_lines.append(f"# recreate with: {recreate}")
    header = "\n".join(header_lines)

    content = f"{header}\n\n{body}"

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding=ENCODING)
    logger.info("Environment snapshot written to %s", path)


def _get_environment_info() -> dict[str, str]:
    """
    Collect metadata describing the current Python environment.
    """
    info = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }

    return info


def _conda_prefix() -> Optional[str]:
    """
    Return the prefix of the conda environment ClinicaDL runs in, if any.

    Returns
    -------
    Optional[str]
        The value of ``CONDA_PREFIX`` when it is set and matches the running
        interpreter (:py:data:`sys.prefix`), meaning ClinicaDL is executed from
        within that conda environment. ``None`` otherwise (plain venv, system
        Python, or a conda install whose base differs from the active env).
    """
    prefix = os.environ.get("CONDA_PREFIX")
    if not prefix:
        return None

    if os.path.normpath(prefix) != os.path.normpath(sys.prefix):
        logger.debug(
            "CONDA_PREFIX (%s) does not match the running interpreter (%s); "
            "falling back to a pip snapshot.",
            prefix,
            sys.prefix,
        )
        return None

    return prefix


def _export_conda_environment(prefix: str, portable: bool = False) -> str:
    """
    Capture the conda environment as an ``environment.yml`` snapshot.

    Runs ``conda env export`` on the environment at ``prefix``. The output pins
    conda packages at the top level and pip-installed packages in a nested
    ``pip:`` block, so it is a complete description of the environment.

    Parameters
    ----------
    prefix : str
        Prefix (path) of the conda environment to export.
    portable : bool, default=False
        If ``True``, export with ``--no-builds --ignore-channels`` to maximize
        the chances of solving on another machine. If ``False``, keep build
        strings and channels for the most faithful same-machine reproduction.
    """
    conda = os.environ.get("CONDA_EXE", "conda")
    command: list[str] = [conda, "env", "export", "-p", prefix]
    if portable:
        command += ["--no-builds", "--ignore-channels"]

    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding=ENCODING,
        )
    except FileNotFoundError as exc:  # conda is not on PATH
        raise RuntimeError(
            f"Could not run '{conda}' to export the conda environment."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"'conda env export' failed with exit code {exc.returncode}:\n{exc.stderr}"
        ) from exc

    return completed.stdout


def _freeze_packages() -> str:
    """
    Capture the installed packages as a ``pip freeze`` snapshot.

    Runs ``pip freeze`` for the interpreter currently running ClinicaDL
    (:py:data:`sys.executable`). The output pins every installed distribution
    to an exact version and is directly usable with ``pip install -r``.

    Returns
    -------
    str
        The ``pip freeze`` output, one ``name==version`` requirement per line.

    Notes
    -----
    A subprocess call to ``pip`` is used rather than :py:mod:`importlib.metadata`
    because it preserves the install *source* of each package: editable installs
    (``-e``), VCS checkouts and direct URLs are kept in a re-installable form,
    whereas reading metadata would flatten them to bare ``name==version`` pins.
    """
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=True,
            capture_output=True,
            text=True,
            encoding=ENCODING,
        )
    except FileNotFoundError as exc:  # pip is missing entirely
        raise RuntimeError(
            f"Could not run 'pip' for the interpreter {sys.executable}."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"'pip freeze' failed with exit code {exc.returncode}:\n{exc.stderr}"
        ) from exc

    return completed.stdout
