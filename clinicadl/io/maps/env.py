from pathlib import Path

from clinicadl.utils.dictionary.suffixes import TXT, YML
from clinicadl.utils.dictionary.words import (
    ENVIRONMENT,
    PORTABLE,
)

from ..base import Directory
from ..utils import mandatory


class EnvDir(Directory):
    @property
    @mandatory
    def environment_txt(self) -> Path:
        return (self.path / ENVIRONMENT).with_suffix(TXT)

    @property
    def environment_yml(self) -> Path:
        return (self.path / ENVIRONMENT).with_suffix(YML)

    @property
    def environment_portable_yml(self) -> Path:
        return (self.path / "_".join([ENVIRONMENT, PORTABLE])).with_suffix(YML)
