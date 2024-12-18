import json
import subprocess
from datetime import datetime
from logging import getLogger
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from pydantic import BaseModel

from clinicadl.dataset.preprocessing import BasePreprocessing
from clinicadl.dataset.readers import CapsReader
from clinicadl.metrics.old_metrics.utils import check_selection_metric
from clinicadl.model.clinicadl_model import ClinicaDLModel
from clinicadl.networks.config import NetworkConfig
from clinicadl.networks.factory import get_network_from_config
from clinicadl.splitter.split_utils import print_description_log
from clinicadl.transforms.extraction import Extraction
from clinicadl.utils.exceptions import MAPSError
from clinicadl.utils.iotools.data_utils import load_data_test
from clinicadl.utils.iotools.utils import path_decoder, path_encoder

logger = getLogger("clinicadl.experiment_manager")


class ExperimentManager:
    def __init__(self, maps_path: Path, overwrite: bool) -> None:
        """TO COMPLETE"""
        if maps_path.is_dir() and not overwrite:
            raise ValueError(
                f"Directory {maps_path} already exists. Use overwrite=True to overwrite."
            )

        self.df_index = ["participant_id", "session_id"]
        self.maps_path = maps_path
        self.maps_json = self.maps_path / "maps.json"
        self.overwrite = overwrite

    def split_dir(self, split: int) -> Path:
        return self.maps_path / f"split-{split}"

    def best_metric_dir(self, split: int, selection_metric: str) -> Path:
        return self.split_dir(split) / f"best-{selection_metric}"

    def data_group_dir(
        self, split: int, selection_metric: str, data_group: str
    ) -> Path:
        return self.best_metric_dir(split, selection_metric) / data_group

    def prediction_tsv(
        self, split: int, selection_metric: str, data_group: str, mode: str
    ) -> Path:
        return (
            self.data_group_dir(split, selection_metric, data_group)
            / f"{data_group}_{mode}_level_prediction.tsv"
        )

    def description_log(
        self, split: int, selection_metric: str, data_group: str
    ) -> Path:
        return (
            self.data_group_dir(split, selection_metric, data_group) / "description.log"
        )

    def train_val_tsv(self) -> Path:
        return self.maps_path / "groups" / "train+validation.tsv"

    def information_log(self) -> Path:
        return self.maps_path / "information.log"

    def get_info_from_json(
        self,
    ) -> tuple[PreprocessingConfig, Extraction, CapsReader, ClinicaDLModel]:
        """Reads the maps.json file and returns its content."""  # I don't know if this is a useful function

        if self.maps_json.is_file():
            with self.maps_json.open(mode="r") as file:
                dict_ = json.load(file, object_hook=path_decoder)
        else:
            raise FileNotFoundError(f"maps.json file not found in {self.maps_json}.")

        preprocessing, extraction = get_preprocessing_and_mode_from_parameters(
            dict_
        )  # function defined in another PR

        caps_reader = CapsReader(caps_directory=dict_["caps_directory"])

        clinicadl_model = get_clinicadl_model_from_parameters(
            dict_
        )  # function to define in another PR

        return (
            preprocessing,
            extraction,
            caps_reader,
            clinicadl_model,
        )  # do we need to return other things ? like a trainConfig or something

    def print_description_log(
        self,
        split: int,
        selection_metric: str,
        data_group: str,
    ):
        """
        Print the description log associated to a prediction or interpretation.

        Args:
            data_group (str): name of the data group used for the task.
            split (int): Index of the split used for training.
            selection_metric (str): Metric used for best weights selection.
        """
        with self.description_log(split, selection_metric, data_group).open(
            mode="r"
        ) as f:
            content = f.read()

    def _get_prediction(
        self,
        data_group: str,
        split: int = 0,
        selection_metric: str = "loss",
        mode: str = "image",  # TODO : need to change this to an ExtractionConfig
        verbose: bool = False,  # TODO: do we remove verbose argument everywhere ?
    ):
        """
        Get the individual predictions for each participant corresponding to one group
        of participants identified by its data group.

        Args:
            data_group (str): name of the data group used for the prediction task.
            split (int): Index of the split used for training.
            selection_metric (str): Metric used for best weights selection.
            mode (str): level of the prediction.
            verbose (bool): if True will print associated prediction.log.
        Returns:
            (DataFrame): Results indexed by columns 'participant_id' and 'session_id' which
            identifies the image in the BIDS / CAPS.
        """
        selection_metric = check_selection_metric(
            self.maps_path, split, selection_metric
        )
        if verbose:
            self.print_description_log(split, selection_metric, data_group)

        if not self.data_group_dir(
            split=split, selection_metric=selection_metric, data_group=data_group
        ).is_dir():
            raise MAPSError(
                f"No prediction corresponding to data group {data_group} was found."
            )
        df = pd.read_csv(
            self.prediction_tsv(
                split=split,
                selection_metric=selection_metric,
                data_group=data_group,
                mode=mode,
            ),
            sep="\t",
        )
        df.set_index(self.df_index, inplace=True, drop=True)
        return df

    def _write_requirements_version(self):
        """Writes the environment.txt file."""
        logger.debug("Writing requirement version...")
        try:
            env_variables = subprocess.check_output("pip freeze", shell=True).decode(
                "utf-8"
            )
            with (self.maps_path / "environment.txt").open(mode="w") as file:
                file.write(env_variables)
        except subprocess.CalledProcessError:
            logger.warning(
                "You do not have the right to execute pip freeze. Your environment will not be written"
            )

    def _write_parameters(self, config: BaseModel, verbose=True):
        """Add config parameters in the JSON file."""

        logger.debug("Writing parameters...")
        # save to json file

        if verbose:
            logger.info(f"Path of json file: {self.maps_json}")

        with self.maps_json.open(mode="a") as json_file:
            json.dump(
                config.model_dump_json(),
                json_file,
                skipkeys=True,
                indent=4,
                default=path_encoder,
            )

    def _write_information(
        self, network_config: NetworkConfig
    ):  # from model directly ?
        """
        Writes model architecture of the MAPS in MAPS root.
        """

        model, _ = get_network_from_config(network_config)

        with self.information_log().open(mode="w") as f:
            f.write(f"- Date :\t{datetime.now().strftime('%d %b %Y, %H:%M:%S')}\n\n")
            f.write(f"- Path :\t{self.maps_path}\n\n")
            # f.write("- Job ID :\t{}\n".format(os.getenv('SLURM_JOBID')))
            f.write(f"- Model :\t{model.layers}\n\n")

        del model

    def _write_training_data(
        self,
        tsv_path: Path,
        diagnoses: list[str],
        multi_cohort: bool = False,
        transfer_path: Optional[Path] = None,
    ):
        """Writes the TSV file containing the participant and session IDs used for training."""
        logger.debug("Writing training data...")

        train_df = load_data_test(
            tsv_path,
            diagnoses,
            baseline=False,
            multi_cohort=multi_cohort,
        )
        train_df = train_df[self.df_index]
        if transfer_path:
            transfer_train_path = transfer_path / "groups" / "train+validation.tsv"
            transfer_train_df = pd.read_csv(transfer_train_path, sep="\t")
            transfer_train_df = transfer_train_df[self.df_index]
            train_df = pd.concat([train_df, transfer_train_df])
            train_df.drop_duplicates(inplace=True)
        train_df.to_csv(self.train_val_tsv(), sep="\t", index=False)

    @staticmethod
    def write_description_log(
        log_dir: Path,
        data_group,
        caps_dict,
        df,
    ):
        """
        Write description log file associated to a data group.

        Args:
            log_dir (str): path to the log file directory.
            data_group (str): name of the data group used for the task.
            caps_dict (dict[str, str]): Dictionary of the CAPS folders used for the task
            df (pd.DataFrame): DataFrame of the meta-data used for the task.
        """
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "description.log"
        with log_path.open(mode="w") as f:
            f.write(f"Prediction {data_group} group - {datetime.now()}\n")
            f.write(f"Data loaded from CAPS directories: {caps_dict}\n")
            f.write(f"Number of participants: {df.participant_id.nunique()}\n")
            f.write(f"Number of sessions: {len(df)}\n")
