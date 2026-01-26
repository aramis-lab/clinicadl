from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any


class MapsSummary:
    """
    To generate a summary of the :term:`MAPS` directory.

    It contains:

    - a header;
    - a "Training" section;
    - a "Test" section;
    - a "Prediction" section.

    Parameters
    ----------
    file : Path
        The summary file.
    """

    def __init__(self, file: Path):
        self._header = _HeaderSection(file)
        self._training = _TrainingSection(file)
        self._test = _TestSection(file)
        self._prediction = _PredictionSection(file)

    def create(self) -> None:
        """
        Creates a new summary file.
        """
        self._header.create()

    def add_training_split(self, split_idx: int) -> None:
        """
        Adds a split on which training was performed in the "Training" section.

        Parameters
        ----------
        split_idx : int
            The index of the split.
        """
        self._training.add_item(split_idx)

    def add_test_group(self, group_name: str) -> None:
        """
        Adds a group on which test was performed in the "Test" section.

        Parameters
        ----------
        group_name : str
            The name of the group.
        """
        self._test.add_item(group_name)

    def add_prediction_group(self, group_name: str) -> None:
        """
        Adds a group on which prediction was performed in the "Prediction" section.

        Parameters
        ----------
        group_name : str
            The name of the group.
        """
        self._prediction.add_item(group_name)


class _Section(ABC):
    """
    A Python object representing a section in the summary file.
    """

    HEADER: str
    DATE_PREFIX = "Date: "
    PATH_PREFIX = "Path: "

    def __init__(self, file: Path):
        self.file = file

    @abstractmethod
    def create(self) -> None:
        """
        To create the section.
        """

    def _write(self, text: str) -> None:
        """To write in the summary file."""
        with self.file.open(mode="w") as f:
            f.write(text)

    def _read(self) -> str:
        """To read the summary file."""
        with self.file.open(mode="r") as f:
            return f.read()

    @classmethod
    def _datetime(cls) -> str:
        """To get current date and time."""
        return cls.DATE_PREFIX + datetime.now().strftime("%Y %b %d, %H:%M:%S")


class _HeaderSection(_Section):
    HEADER = "==================== MAPS summary ===================="

    def create(self) -> None:
        summary = self.HEADER
        summary += "\n\n"
        summary += self._datetime()
        summary += "\n"
        summary += self.PATH_PREFIX + str(self.file.parent.resolve())
        summary += "\n\n"

        self._write(summary)


class _ListSection(_Section):
    """
    To read a section like:

    <HEADER>

    <KEY>
        - <item>
        - <item>

    <End of section>
    """

    KEY: str
    BULLET_POINT = "-"
    DATE = False

    def __init__(self, file: Path):
        super().__init__(file)
        self.items: set[str] = set()

    def add_item(self, item: Any) -> None:
        """
        Adds an item in the enumeration.

        Parameters
        ----------
        item : str
            The item to add.
        """
        content = self._read()
        if self.HEADER not in content:
            self.create()

        item = str(item)
        self._read_items()

        if item in self.items:
            self._replace_item(item)
        else:
            self._add_item(item)

    def create(self) -> None:
        section_header = self.HEADER
        section_header += "\n\n"
        section_header += self.KEY
        section_header += "\n\n"

        with open(self.file, "a") as f:
            f.write(section_header)

    def _read_items(self) -> None:
        """
        Reads the items already in the enumeration.
        """
        in_section = False
        in_list = False

        with open(self.file, "r") as f:
            for line in f:
                if line.strip() == self.HEADER:
                    in_section = True
                elif in_section and line.strip().startswith(self.KEY):
                    in_list = True
                elif in_list and line.strip().startswith(self.BULLET_POINT):
                    item = line.replace(self.BULLET_POINT, "").strip()
                    self.items.add(item)
                elif in_list and not line.strip():
                    break

    def _add_item(self, item: str) -> None:
        """
        Adds an item in the enumeration if does not already exist.
        """
        out = []
        in_section = False
        in_list = False

        with open(self.file, "r") as f:
            for line in f:
                if line.strip() == self.HEADER:
                    in_section = True
                elif in_section and line.strip().startswith(self.KEY):
                    in_list = True
                elif in_list and not line.strip():
                    out.append(self._new_bullet(item))
                    if self.DATE:
                        out.append(self._datetime())
                    in_list = False

                out.append(line)

        with open(self.file, "w") as f:
            f.writelines(out)

    def _replace_item(self, item: str) -> None:
        """
        Updates an item if it already exists.
        """
        if not self.DATE:
            return

        out = []
        in_target_item = False

        with open(self.file, "r") as f:
            for line in f:
                if line.strip() == self.BULLET_POINT + " " + item:
                    in_target_item = True
                elif in_target_item and line.strip().startswith(self.DATE_PREFIX):
                    out.append(self._datetime())
                    in_target_item = False
                    continue

                out.append(line)

        with open(self.file, "w") as f:
            f.writelines(out)

    def _new_bullet(self, item: str) -> str:
        """To create a new bullet point in the enumeration."""
        return " " * 3 + self.BULLET_POINT + " " + item + "\n"

    @classmethod
    def _datetime(cls) -> str:
        return " " * 6 + super()._datetime() + "\n"


class _TrainingSection(_ListSection):
    HEADER = "---------------------- Training ----------------------"
    KEY = "Splits"
    DATE = True


class _TestSection(_ListSection):
    HEADER = "------------------------ Test ------------------------"
    KEY = "Groups"


class _PredictionSection(_ListSection):
    HEADER = "--------------------- Prediction ---------------------"
    KEY = "Groups"
