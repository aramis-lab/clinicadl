import os
import re
import sys
from datetime import timedelta
from functools import wraps
from pathlib import Path
from time import monotonic
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from clinicadl.io.maps import Maps
from clinicadl.utils.dictionary.suffixes import TSV
from clinicadl.utils.exceptions import add_note


def setup_ddp(rank: int, world_size: int, port: int, use_cuda: bool = True) -> None:
    """
    Set up a process group, using Gloo on CPU or NCCL on CUDA.
    """
    if use_cuda:
        assert torch.cuda.device_count() >= world_size
        torch.cuda.set_device(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    backend = "nccl" if use_cuda else "gloo"
    kwargs = {} if use_cuda else {"timeout": timedelta(seconds=30)}
    if not use_cuda and sys.platform == "win32":
        # Windows PyTorch wheels may not include the libuv TCP store backend.
        kwargs["init_method"] = "env://?use_libuv=0"
    dist.init_process_group(backend, rank=rank, world_size=world_size, **kwargs)


def ddp_cleanup() -> None:
    """
    Stops DDP.
    """
    if dist.is_initialized():
        dist.destroy_process_group()


def ddp_wrapper(func: callable) -> callable:
    """
    Turns a function into a function compatible
    with DDP multiprocessing.
    """

    @wraps(func)
    def wrapped(rank, world_size, port, use_cuda=True, *args, **kwargs):
        try:
            setup_ddp(rank, world_size, port, use_cuda=use_cuda)
            func(rank, *args, **kwargs)
        finally:
            ddp_cleanup()

    return wrapped


def ddp_test(
    func: callable, world_size: int, *, use_cuda: bool = True, timeout: float = 120
) -> None:
    """
    Launch workers. CPU mode is bounded by ``timeout`` seconds, including startup.
    """
    port = np.random.randint(10000, 20000)
    if use_cuda:
        mp.spawn(func, args=(world_size, port), nprocs=world_size, join=True)
        return

    context = mp.spawn(
        func, args=(world_size, port, False), nprocs=world_size, join=False
    )
    deadline = monotonic() + timeout
    try:
        while not context.join(timeout=min(1, max(0, deadline - monotonic()))):
            if monotonic() >= deadline:
                raise TimeoutError(
                    "CPU distributed test workers did not finish in time."
                )
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()


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
                add_note(e, f"Error raised when comparing {file} and {ref_file}")
                raise


def _compare_any_file(file: Path, ref_file: Path) -> None:
    """
    To compare any files that can be found in a MAPS.
    """
    content = Maps.open_file(file)
    ref_content = Maps.open_file(ref_file)

    if file.name.startswith("environment"):
        return

    elif file.name == "computational.tsv":
        _soft_compare_df(content, ref_content)

    elif file.suffix == TSV:
        pd.testing.assert_frame_equal(content, ref_content)
    else:
        _compare_anything(content, ref_content)


def _compare_anything(content: Any, ref_content: Any) -> None:
    """
    To compare any information that can be found in a MAPS.
    """
    if isinstance(content, dict) and isinstance(ref_content, dict):
        assert set(content.keys()) == set(ref_content.keys()), (
            f"OUT: {content.keys()}, REF: {ref_content.keys()}"
        )
        for key in content:
            try:
                _compare_anything(content[key], ref_content[key])
            except AssertionError as e:
                add_note(e, f"Error raised when comparing '{key}'")
                raise
    elif isinstance(content, list) and isinstance(ref_content, list):
        assert len(content) == len(ref_content)
        try:
            content = sorted(content)
            ref_content = sorted(ref_content)
        except TypeError:
            pass
        for c, ref_c in zip(content, ref_content):
            _compare_anything(c, ref_c)
    elif isinstance(content, torch.Tensor) and isinstance(ref_content, torch.Tensor):
        torch.testing.assert_close(content, ref_content)
    else:
        if isinstance(content, str) and isinstance(ref_content, str):
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
OBJECT_ADRESS_PATTERN = r"\s+at 0x[0-9A-Fa-f]+"
DATE_PATTERN = r"^(\s*Date:\s*).*$|\b\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\b"
THROUGHPUT_PATTERN = r"^(\s*Throughput:\s*).*$"
NUMBER_PATTERN = r"\b\d*\.\d+(?:[eE][+-]?\d+)?\b|\b\d+[eE][+-]?\d+\b"
TRACE_BACK_PATTERN = r"Traceback[\s\S]*?(?=torch\.OutOfMemoryError)"


def _normalize_str(text: str) -> str:
    """
    To remove dates and numerical values in a file.
    """
    normalized = []
    for line in text.splitlines():
        line = re.sub(THROUGHPUT_PATTERN, r"\1<throughput>", line)
        line = re.sub(DATE_PATTERN, r"\1<date>", line)
        line = re.sub(OBJECT_ADRESS_PATTERN, "<hash>", line)
        line = re.sub(
            NUMBER_PATTERN,
            lambda m: _replace_with_same_length(m, "x"),
            line,
        )
        line = re.sub(PATH_PATTERN, "<path>", line)

        normalized.append(line)

    file = "\n".join(normalized)
    file = re.sub(TRACE_BACK_PATTERN, "<traceback>", file)

    return file


def _replace_with_same_length(match, char="x"):
    """
    Return a replacement string with the same length as the match.
    """
    return char * len(match.group(0))
