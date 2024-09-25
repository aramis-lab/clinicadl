import json
import subprocess
from datetime import datetime
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import torch
import torch.distributed as dist
from torch.amp import autocast

from clinicadl.caps_dataset.caps_dataset_utils import read_json
from clinicadl.caps_dataset.data import (
    return_dataset,
)
from clinicadl.metrics.metric_module import MetricModule
from clinicadl.metrics.utils import (
    check_selection_metric,
    find_selection_metrics,
)
from clinicadl.predict.utils import get_prediction
from clinicadl.trainer.tasks_utils import (
    ensemble_prediction,
    evaluation_metrics,
    generate_label_code,
    output_size,
    test,
    test_da,
)
from clinicadl.transforms.config import TransformsConfig
from clinicadl.utils import cluster
from clinicadl.utils.computational.ddp import DDP, init_ddp
from clinicadl.utils.exceptions import (
    ClinicaDLArgumentError,
    ClinicaDLConfigurationError,
    MAPSError,
)
from clinicadl.utils.iotools.maps_manager_utils import (
    add_default_values,
)
from clinicadl.utils.iotools.utils import path_encoder

logger = getLogger("clinicadl.maps_manager")


class InOutManager:
    def __init__(self, maps_path: Path, config):
        self.maps_path = maps_path

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

    def _write_training_data(
        self,
        tsv_path: Path,
        diagnoses: List[str],
        multi_cohort: bool,
        transfer_path: Optional[Path] = None,
    ):
        """Writes the TSV file containing the participant and session IDs used for training."""
        logger.debug("Writing training data...")
        from clinicadl.utils.iotools.data_utils import load_data_test

        train_df = load_data_test(
            tsv_path,  # self.tsv_path,
            diagnoses,  # self.diagnoses,
            baseline=False,
            multi_cohort=multi_cohort,  # self.multi_cohort,
        )
        train_df = train_df[["participant_id", "session_id"]]
        if transfer_path:  # self.transfer_path:
            transfer_train_path = transfer_path / "groups" / "train+validation.tsv"
            transfer_train_df = pd.read_csv(transfer_train_path, sep="\t")
            transfer_train_df = transfer_train_df[["participant_id", "session_id"]]
            train_df = pd.concat([train_df, transfer_train_df])
            train_df.drop_duplicates(inplace=True)
        train_df.to_csv(
            self.maps_path / "groups" / "train+validation.tsv", sep="\t", index=False
        )

    def _write_train_val_groups(
        self, label: str, split_name: str, caps_directory: Path, multi_cohort: bool
    ):
        """Defines the training and validation groups at the initialization"""
        logger.debug("Writing training and validation groups...")
        split_manager = self._init_split_manager()
        for split in split_manager.split_iterator():
            for data_group in ["train", "validation"]:
                df = split_manager[split][data_group]
                group_path = (
                    self.maps_path / "groups" / data_group / f"{split_name}-{split}"
                )
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

    def _write_information(self, architecture, parameters):
        """
        Writes model architecture of the MAPS in MAPS root.
        """
        from datetime import datetime

        import clinicadl.network as network_package

        model_class = getattr(network_package, architecture)
        args = list(
            model_class.__init__.__code__.co_varnames[
                : model_class.__init__.__code__.co_argcount
            ]
        )
        args.remove("self")
        kwargs = dict()
        for arg in args:
            kwargs[arg] = parameters[arg]
        kwargs["gpu"] = False

        model = model_class(**kwargs)

        file_name = "information.log"

        with (self.maps_path / file_name).open(mode="w") as f:
            f.write(f"- Date :\t{datetime.now().strftime('%d %b %Y, %H:%M:%S')}\n\n")
            f.write(f"- Path :\t{self.maps_path}\n\n")
            # f.write("- Job ID :\t{}\n".format(os.getenv('SLURM_JOBID')))
            f.write(f"- Model :\t{model.layers}\n\n")

        del model

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
