import os
import re
from functools import wraps
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from clinicadl.io import Maps
from clinicadl.utils.dictionary.suffixes import PT, TSV


def setup_ddp(rank: int, world_size: int, port: int) -> None:
    """
    Sets up DDP. Expects of course GPUs.
    """
    assert torch.cuda.device_count() >= world_size
    torch.cuda.set_device(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    backend = "nccl"
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def ddp_cleanup() -> None:
    """
    Stops DDP.
    """
    dist.destroy_process_group()


def ddp_wrapper(func: callable) -> callable:
    """
    Turns a function into a function compatible
    with DDP multiprocessing.
    """

    @wraps(func)
    def wrapped(rank, world_size, port, *args, **kwargs):
        try:
            setup_ddp(rank, world_size, port)
            func(rank, *args, **kwargs)
        finally:
            ddp_cleanup()

    return wrapped


def ddp_test(func: callable, world_size: int) -> None:
    """
    Selects and random port and launch parallelism.
    """
    port = np.random.randint(10000, 20000)
    mp.spawn(func, args=(world_size, port), nprocs=world_size, join=True)


def compare_maps_dir(
    out_dir: Path, ref_dir: Path, except_: Optional[list[str | Path]] = None
) -> None:
    """
    To compare any directory of two MAPS.
    """
    import os

    if except_ is None:
        except_ = []
    for i, path in enumerate(except_):
        except_[i] = Path(path)

    assert len(list(Path(out_dir).iterdir())) > 0
    assert len(list(Path(ref_dir).iterdir())) > 0

    for (root, dirs, files), (ref_root, ref_dirs, ref_files) in zip(
        os.walk(out_dir), os.walk(ref_dir)
    ):
        dirs.sort()  # will affect next iteration
        ref_dirs.sort()

        ref_root = Path(ref_root)
        for dir_ in dirs:
            if (ref_root / dir_).relative_to(ref_dir) in except_:
                dirs.remove(dir_)
                ref_dirs.remove(dir_)
        for f in files:
            if (ref_root / f).relative_to(ref_dir) in except_:
                files.remove(f)
                ref_files.remove(f)

        for file, ref_file in zip(sorted(files), sorted(ref_files)):
            assert file == ref_file, f"Comparing {file} and {ref_file}"
            try:
                _compare_any_file(
                    file := Path(root) / file, ref_file := Path(ref_root) / ref_file
                )
            except AssertionError as e:
                e.add_note(f"Error raised when comparing {file} and {ref_file}")
                raise


def _compare_any_file(file: Path, ref_file: Path) -> None:
    """
    To compare any files that can be found in a MAPS.
    """
    content = Maps.open_file(file)
    ref_content = Maps.open_file(ref_file)

    if file.name == "environment.txt":
        return

    elif file.name == "summary.log":
        content = _normalize_file(content)
        ref_content = _normalize_file(ref_content)

    elif file.name == "computational.tsv":
        _soft_compare_df(content, ref_content)

    elif file.suffix == TSV:
        pd.testing.assert_frame_equal(content, ref_content)
    elif file.suffix == PT:
        torch.testing.assert_close(content, ref_content)
    else:
        _compare_anything(content, ref_content)


def _compare_anything(content: Any, ref_content: Any) -> None:
    """
    To compare any information that can be found in a MAPS.
    """
    if isinstance(content, dict) and isinstance(ref_content, dict):
        assert set(content.keys()) == set(ref_content.keys())
        for key in content:
            try:
                _compare_anything(content[key], ref_content[key])
            except AssertionError as e:
                e.add_note(f"Error raised when comparing '{key}'")
                raise
    else:
        if isinstance(content, list) and isinstance(ref_content, list):
            try:
                content = sorted(content)
                ref_content = sorted(ref_content)
            except TypeError:
                pass
        elif isinstance(content, str) and isinstance(ref_content, str):
            content = _normalize_str(content)
            ref_content = _normalize_str(ref_content)

        assert content == ref_content, f"\nOUT:\n {content}\n" + f"REF:\n {ref_content}"


def _soft_compare_df(df1: pd.DataFrame, df2: pd.DataFrame) -> None:
    """
    To compare only the number of values of each columns.
    """
    assert (df1.columns == df2.columns).all()
    for column in df1.columns:
        assert len(df1[column].dropna()) == len(df2[column].dropna())


PATH_PATTERN = r'(?:[A-Za-z]:\\[^ \n\r\t]*)|(?:/[^\s"\']+)'
LOG_DATE_PATTERN = r"\b\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\b"
DATE_PATTERN = re.compile(r"^(\s*Date:\s*).*$")
THROUGHPUT_PATTERN = re.compile(r"^(\s*Throughput:\s*).*$")
NUMBER_PATTERN = re.compile(
    r"\b\d+\.\d+(?:[eE][+-]?\d+)?\b(?:\s*±\s*\d+\.\d+(?:[eE][+-]?\d+)?)?"
)


def _normalize_str(str_: str) -> str:
    """
    To remove dates and paths in a string.
    """
    str_ = re.sub(PATH_PATTERN, "<path>", str_)
    str_ = re.sub(LOG_DATE_PATTERN, "<date>", str_)

    return str_


def _normalize_file(text: str) -> str:
    """
    To remove dates and numerical values in a file.
    """
    normalized = []
    for line in text.splitlines():
        if THROUGHPUT_PATTERN.match(line):
            line = THROUGHPUT_PATTERN.sub(r"\1<throughput>", line.rstrip("\n"))
        elif DATE_PATTERN.match(line):
            line = DATE_PATTERN.sub(r"\1<date>", line.rstrip("\n"))
        else:
            line = NUMBER_PATTERN.sub(
                lambda m: _replace_with_same_length(m, "x"), line.rstrip("\n")
            )

        normalized.append(line)

    return "\n".join(normalized)


def _replace_with_same_length(match, char="x"):
    """
    Return a replacement string with the same length as the match.
    """
    return char * len(match.group(0))
