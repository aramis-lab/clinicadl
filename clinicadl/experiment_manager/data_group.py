
import pandas as pd

from typing import List, Optional
from pathlib import Path

from clinicadl.data.utils import tsv_to_df
from clinicadl.splitter.split import Split
from clinicadl.utils.config import ClinicaDLConfig

from clinicadl.dictionary.words import GROUPS, DATA, MAPS, TRAIN, VALIDATION, SPLIT
from clinicadl.dictionary.suffixes import TSV, JSON



TRAIN_VAL = [TRAIN, VALIDATION]


class DataGroup(ClinicaDLConfig):
    maps_path: Path
    name: str
    split: Optional[int]

    @property
    def data_tsv(self)-> Path:
        return self.group_split_dir / (DATA + TSV)
    
    @property
    def maps_json(self)-> Path:
        return self.group_split_dir / (MAPS + JSON)
        
    @property
    def group_dir(self) -> Path:
        return self.maps_path / GROUPS / self.name
    
    @property
    def group_split_dir(self) -> Path:
        if self.name in TRAIN_VAL:
            return self.group_dir / (SPLIT + "-" + str(self.split))
        else:
            return self.group_dir

    def get_data(self)-> pd.DataFrame:
        return tsv_to_df(self.data_tsv)
