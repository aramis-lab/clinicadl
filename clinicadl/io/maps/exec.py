from pathlib import Path
from time import gmtime, strftime

from clinicadl.utils.dictionary.suffixes import LOG
from clinicadl.utils.dictionary.words import DEBUG, ERROR, INFO, RUN

from ..base import Directory
from .utils import CollectionOfDirs


class RunDir(Directory):
    @property
    def info(self) -> Path:
        return (self.path / INFO).with_suffix(LOG)

    @property
    def error(self) -> Path:
        return (self.path / ERROR).with_suffix(LOG)

    @property
    def debug(self) -> Path:
        return (self.path / DEBUG).with_suffix(LOG)


class ExecDir(CollectionOfDirs[RunDir, str]):
    _item_key = RUN
    _dir_type = RunDir

    def __init__(self, path: Path):
        super().__init__(path)
        self._runs: dict[str, RunDir] = {}

    @property
    def runs(self) -> dict[str, RunDir]:
        return self._runs

    @property
    def runs_list(self) -> list[str]:
        return self._items_list

    def create_run(self, process_called: str) -> str:
        datetime = strftime("%Y_%m_%d_%H_%M_%S", gmtime())
        run_name = f"{process_called}_{datetime}"
        self._create_item(run_name, overwrite=True, exist_ok=True)

        return run_name
