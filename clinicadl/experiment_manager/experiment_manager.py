import json
import subprocess
from datetime import datetime
from logging import getLogger
from pathlib import Path
from typing import Optional

import pandas as pd
from pydantic import BaseModel

from clinicadl.dataset.data_utils import load_data_test
from clinicadl.networks.config import NetworkConfig
from clinicadl.networks.factory import get_network_from_config
from clinicadl.splitter.kfold import KFolder
from clinicadl.utils.iotools.utils import path_encoder

logger = getLogger("clinicadl.experiment_manager")


class ExperimentManager:
    def __init__(self, maps_path: Path, overwrite: bool) -> None:
        """TO COMPLETE"""
        if maps_path.is_dir() and not overwrite:
            raise ValueError(
                f"Directory {maps_path} already exists. Use overwrite=True to overwrite."
            )

        self.maps_path = maps_path

    @classmethod
    def from_existing_maps(cls, existing_maps_path: Path, overwrite: bool):
        """TO COMPLETE"""

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
        """Write JSON files of parameters."""

        logger.debug("Writing parameters...")
        # save to json file
        maps_json = self.maps_path / "maps.json"

        if verbose:
            logger.info(f"Path of json file: {maps_json}")

        with maps_json.open(mode="a") as json_file:
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
        from datetime import datetime

        model, _ = get_network_from_config(network_config)

        file_name = "information.log"

        with (self.maps_path / file_name).open(mode="w") as f:
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
        train_df = train_df[["participant_id", "session_id"]]
        if transfer_path:
            transfer_train_path = transfer_path / "groups" / "train+validation.tsv"
            transfer_train_df = pd.read_csv(transfer_train_path, sep="\t")
            transfer_train_df = transfer_train_df[["participant_id", "session_id"]]
            train_df = pd.concat([train_df, transfer_train_df])
            train_df.drop_duplicates(inplace=True)
        train_df.to_csv(
            self.maps_path / "groups" / "train+validation.tsv", sep="\t", index=False
        )

    def _write_train_val_groups(
        self,
        split_manager: KFolder,
        label: str,
        caps_directory: Path,
        multi_cohort: bool = False,
    ):
        """Defines the training and validation groups at the initialization"""
        logger.debug("Writing training and validation groups...")

        for split in split_manager.split_iterator():
            for data_group in ["train", "validation"]:
                df = split_manager[split][data_group]
                group_path = self.maps_path / "groups" / data_group / f"split-{split}"
                group_path.mkdir(parents=True, exist_ok=True)

                columns = ["participant_id", "session_id", "cohort"]
                if label is not None:
                    columns.append(label)
                df.to_csv(group_path / "data.tsv", sep="\t", columns=columns)
                self.write_parameters(
                    group_path,
                    {
                        "caps_directory": caps_directory,
                        "multi_cohort": multi_cohort,
                    },
                    verbose=False,
                )

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
