import json
import shutil
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd
import torch
import torch.distributed as dist
from torch.amp import autocast
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from clinicadl.caps_dataset.data import (
    return_dataset,
)
from clinicadl.interpret.config import InterpretConfig
from clinicadl.maps_manager.maps_manager import MapsManager
from clinicadl.metrics.metric_module import MetricModule
from clinicadl.metrics.utils import (
    check_selection_metric,
    find_selection_metrics,
)
from clinicadl.network.network import Network
from clinicadl.predictor.config import PredictConfig
from clinicadl.trainer.tasks_utils import (
    columns,
    compute_metrics,
    generate_label_code,
    generate_test_row,
    get_criterion,
)
from clinicadl.transforms.config import TransformsConfig
from clinicadl.utils.computational.ddp import DDP, cluster
from clinicadl.utils.enum import Task
from clinicadl.utils.exceptions import (
    ClinicaDLArgumentError,
    ClinicaDLDataLeakageError,
    MAPSError,
)

logger = getLogger("clinicadl.predict_manager")
level_list: List[str] = ["warning", "info", "debug"]


class Predictor:
    def __init__(self, _config: Union[PredictConfig, InterpretConfig]) -> None:
        self._config = _config

        from clinicadl.splitter.config import SplitterConfig
        from clinicadl.splitter.splitter import Splitter

        self.maps_manager = MapsManager(_config.maps_manager.maps_dir)
        self._config.adapt_with_maps_manager_info(self.maps_manager)
        print(self._config.data.model_dump())
        tmp = self._config.data.model_dump(
            exclude=set(["preprocessing_dict", "mode", "caps_dict"])
        )
        print(tmp)
        tmp.update(self._config.split.model_dump())
        print(tmp)
        tmp.update(self._config.validation.model_dump())
        print(tmp)
        self.splitter = Splitter(SplitterConfig(**tmp))

    def predict(
        self,
        label_code: Union[str, dict[str, int]] = "default",
    ):
        """Performs the prediction task on a subset of caps_directory defined in a TSV file."""

        group_df = self._config.data.create_groupe_df()
        self._check_data_group(group_df)
        criterion = get_criterion(
            self.maps_manager.network_task, self.maps_manager.loss
        )

        for split in self.splitter.split_iterator():
            logger.info(f"Prediction of split {split}")
            group_df, group_parameters = self.get_group_info(
                self._config.maps_manager.data_group, split
            )
            # Find label code if not given
            if self._config.data.is_given_label_code(
                self.maps_manager.label, label_code
            ):
                generate_label_code(
                    self.maps_manager.network_task, group_df, self._config.data.label
                )
            # Erase previous TSV files on master process
            if not self._config.validation.selection_metrics:
                split_selection_metrics = find_selection_metrics(
                    self.maps_manager.maps_path,
                    split,
                )
            else:
                split_selection_metrics = self._config.validation.selection_metrics
            for selection in split_selection_metrics:
                tsv_dir = (
                    self.maps_manager.maps_path
                    / f"split-{split}"
                    / f"best-{selection}"
                    / self._config.maps_manager.data_group
                )
                tsv_pattern = f"{self._config.maps_manager.data_group}*.tsv"
                for tsv_file in tsv_dir.glob(tsv_pattern):
                    tsv_file.unlink()

            self._config.data.check_label(self.maps_manager.label)
            if self.maps_manager.multi_network:
                for network in range(self.maps_manager.num_networks):
                    self._predict_single(
                        group_parameters,
                        group_df,
                        self._config.transforms,
                        label_code,
                        criterion,
                        split,
                        split_selection_metrics,
                        network,
                    )
            else:
                self._predict_single(
                    group_parameters,
                    group_df,
                    self._config.transforms,
                    label_code,
                    criterion,
                    split,
                    split_selection_metrics,
                )
            if cluster.master:
                self._ensemble_prediction(
                    self.maps_manager,
                    self._config.maps_manager.data_group,
                    split,
                    self._config.validation.selection_metrics,
                    self._config.data.use_labels,
                    self._config.validation.skip_leak_check,
                )

    def _predict_single(
        self,
        group_parameters,
        group_df,
        transforms,
        label_code,
        criterion,
        split,
        split_selection_metrics,
        network: Optional[int] = None,
    ):
        """_summary_"""

        assert isinstance(self._config, PredictConfig)
        # assert self._config.data.label

        data_test = return_dataset(
            group_parameters["caps_directory"],
            group_df,
            self.maps_manager.preprocessing_dict,
            transforms_config=self._config.transforms,
            multi_cohort=group_parameters["multi_cohort"],
            label_presence=self._config.data.use_labels,
            label=self._config.data.label,
            label_code=(
                self.maps_manager.label_code if label_code == "default" else label_code
            ),
            cnn_index=network,
        )
        test_loader = DataLoader(
            data_test,
            batch_size=(
                self._config.dataloader.batch_size
                if self._config.dataloader.batch_size is not None
                else self.maps_manager.batch_size
            ),
            shuffle=False,
            sampler=DistributedSampler(
                data_test,
                num_replicas=cluster.world_size,
                rank=cluster.rank,
                shuffle=False,
            ),
            num_workers=self._config.dataloader.n_proc
            if self._config.dataloader.n_proc is not None
            else self.maps_manager.n_proc,
        )
        self._test_loader(
            maps_manager=self.maps_manager,
            dataloader=test_loader,
            criterion=criterion,
            data_group=self._config.maps_manager.data_group,
            split=split,
            selection_metrics=split_selection_metrics,
            use_labels=self._config.data.use_labels,
            gpu=self._config.computational.gpu,
            amp=self._config.computational.amp,
            network=network,
        )
        if self._config.maps_manager.save_tensor:
            logger.debug("Saving tensors")
            print("save_tensor")
            self._compute_output_tensors(
                maps_manager=self.maps_manager,
                dataset=data_test,
                data_group=self._config.maps_manager.data_group,
                split=split,
                selection_metrics=self._config.validation.selection_metrics,
                gpu=self._config.computational.gpu,
                network=network,
            )
        if self._config.maps_manager.save_nifti:
            self._compute_output_nifti(
                dataset=data_test,
                split=split,
                network=network,
            )
        if self._config.maps_manager.save_latent_tensor:
            self._compute_latent_tensors(
                dataset=data_test,
                split=split,
                network=network,
            )

    def _compute_latent_tensors(
        self,
        dataset,
        split: int,
        nb_images: Optional[int] = None,
        network: Optional[int] = None,
    ):
        """
        Compute the output tensors and saves them in the MAPS.
        Parameters
        ----------
        dataset : _type_
            wrapper of the data set.
        data_group : _type_
            name of the data group used for the task.
        split : _type_
            split number.
        selection_metrics : _type_
            metrics used for model selection.
        nb_images : _type_ (optional, default=None)
            number of full images to write. Default computes the outputs of the whole data set.
        gpu : _type_ (optional, default=None)
            If given, a new value for the device of the model will be computed.
        network : _type_ (optional, default=None)
            Index of the network tested (only used in multi-network setting).
        """
        for selection_metric in self._config.validation.selection_metrics:
            # load the best trained model during the training
            model, _ = self.maps_manager._init_model(
                transfer_path=self.maps_manager.maps_path,
                split=split,
                transfer_selection=selection_metric,
                gpu=self._config.computational.gpu,
                network=network,
                nb_unfrozen_layer=self.maps_manager.nb_unfrozen_layer,
            )
            model = DDP(
                model,
                fsdp=self.maps_manager.fully_sharded_data_parallel,
                amp=self.maps_manager.amp,
            )
            model.eval()
            tensor_path = (
                self.maps_manager.maps_path
                / f"split-{split}"
                / f"best-{selection_metric}"
                / self._config.maps_manager.data_group
                / "latent_tensors"
            )
            if cluster.master:
                tensor_path.mkdir(parents=True, exist_ok=True)
            dist.barrier()
            if nb_images is None:  # Compute outputs for the whole data set
                nb_modes = len(dataset)
            else:
                nb_modes = nb_images * dataset.elem_per_image
            for i in [
                *range(cluster.rank, nb_modes, cluster.world_size),
                *range(int(nb_modes % cluster.world_size <= cluster.rank)),
            ]:
                data = dataset[i]
                image = data["image"]
                logger.debug(f"Image for latent representation {image}")
                with autocast("cuda", enabled=self.maps_manager.std_amp):
                    _, latent, _ = model.module._forward(
                        image.unsqueeze(0).to(model.device)
                    )
                latent = latent.squeeze(0).cpu().float()
                participant_id = data["participant_id"]
                session_id = data["session_id"]
                mode_id = data[f"{self.maps_manager.mode}_id"]
                output_filename = f"{participant_id}_{session_id}_{self.maps_manager.mode}-{mode_id}_latent.pt"
                torch.save(latent, tensor_path / output_filename)

    @torch.no_grad()
    def _compute_output_nifti(
        self,
        dataset,
        split: int,
        network: Optional[int] = None,
    ):
        """Computes the output nifti images and saves them in the MAPS.
        Parameters
        ----------
        dataset : _type_
            _description_
        data_group : str
            name of the data group used for the task.
        split : int
            split number.
        selection_metrics : list[str]
            metrics used for model selection.
        gpu : bool (optional, default=None)
            If given, a new value for the device of the model will be computed.
        network : int (optional, default=None)
            Index of the network tested (only used in multi-network setting).
        Raises
        --------
        ClinicaDLException if not an image
        """
        import nibabel as nib
        from numpy import eye

        for selection_metric in self._config.validation.selection_metrics:
            # load the best trained model during the training
            model, _ = self.maps_manager._init_model(
                transfer_path=self.maps_manager.maps_path,
                split=split,
                transfer_selection=selection_metric,
                gpu=self._config.computational.gpu,
                network=network,
                nb_unfrozen_layer=self.maps_manager.nb_unfrozen_layer,
            )
            model = DDP(
                model,
                fsdp=self.maps_manager.fully_sharded_data_parallel,
                amp=self.maps_manager.amp,
            )
            model.eval()
            nifti_path = (
                self.maps_manager.maps_path
                / f"split-{split}"
                / f"best-{selection_metric}"
                / self._config.maps_manager.data_group
                / "nifti_images"
            )
            if cluster.master:
                nifti_path.mkdir(parents=True, exist_ok=True)
            dist.barrier()
            nb_imgs = len(dataset)
            for i in [
                *range(cluster.rank, nb_imgs, cluster.world_size),
                *range(int(nb_imgs % cluster.world_size <= cluster.rank)),
            ]:
                data = dataset[i]
                image = data["image"]
                x = image.unsqueeze(0).to(model.device)
                with autocast("cuda", enabled=self.maps_manager.std_amp):
                    output = model(x)
                output = output.squeeze(0).detach().cpu().float()
                # Convert tensor to nifti image with appropriate affine
                input_nii = nib.nifti1.Nifti1Image(
                    image[0].detach().cpu().numpy(), eye(4)
                )
                output_nii = nib.nifti1.Nifti1Image(output[0].numpy(), eye(4))
                # Create file name according to participant and session id
                participant_id = data["participant_id"]
                session_id = data["session_id"]
                input_filename = f"{participant_id}_{session_id}_image_input.nii.gz"
                output_filename = f"{participant_id}_{session_id}_image_output.nii.gz"
                nib.loadsave.save(input_nii, nifti_path / input_filename)
                nib.loadsave.save(output_nii, nifti_path / output_filename)

    def interpret(self):
        """Performs the interpretation task on a subset of caps_directory defined in a TSV file.
        The mean interpretation is always saved, to save the individual interpretations set save_individual to True.
        """
        assert isinstance(self._config, InterpretConfig)

        self._config.adapt_with_maps_manager_info(self.maps_manager)

        if self.maps_manager.multi_network:
            raise NotImplementedError(
                "The interpretation of multi-network framework is not implemented."
            )
        transforms = TransformsConfig(
            normalize=self.maps_manager.normalize,
            data_augmentation=self.maps_manager.data_augmentation,
            size_reduction=self.maps_manager.size_reduction,
            size_reduction_factor=self.maps_manager.size_reduction_factor,
        )
        group_df = self._config.data.create_groupe_df()
        self._check_data_group(group_df)

        assert self._config.split
        for split in self.splitter.split_iterator():
            logger.info(f"Interpretation of split {split}")
            df_group, parameters_group = self.get_group_info(
                self._config.maps_manager.data_group, split
            )
            data_test = return_dataset(
                parameters_group["caps_directory"],
                df_group,
                self.maps_manager.preprocessing_dict,
                transforms_config=transforms,
                multi_cohort=parameters_group["multi_cohort"],
                label_presence=False,
                label_code=self.maps_manager.label_code,
                label=self.maps_manager.label,
            )
            test_loader = DataLoader(
                data_test,
                batch_size=self._config.dataloader.batch_size,
                shuffle=False,
                num_workers=self._config.dataloader.n_proc,
            )
            if not self._config.validation.selection_metrics:
                self._config.validation.selection_metrics = find_selection_metrics(
                    self.maps_manager.maps_path,
                    split,
                )
            for selection_metric in self._config.validation.selection_metrics:
                logger.info(f"Interpretation of metric {selection_metric}")
                results_path = (
                    self.maps_manager.maps_path
                    / f"split-{split}"
                    / f"best-{selection_metric}"
                    / self._config.maps_manager.data_group
                    / f"interpret-{self._config.interpret.name}"
                )
                if (results_path).is_dir():
                    if self._config.interpret.overwrite_name:
                        shutil.rmtree(results_path)
                    else:
                        raise MAPSError(
                            f"Interpretation name {self._config.interpret.name} is already written. "
                            f"Please choose another name or set overwrite_name to True."
                        )
                results_path.mkdir(parents=True)
                model, _ = self.maps_manager._init_model(
                    transfer_path=self.maps_manager.maps_path,
                    split=split,
                    transfer_selection=selection_metric,
                    gpu=self._config.computational.gpu,
                )
                interpreter = self._config.interpret.get_method()(model)
                cum_maps = [0] * data_test.elem_per_image
                for data in test_loader:
                    images = data["image"].to(model.device)
                    map_pt = interpreter.generate_gradients(
                        images,
                        self._config.interpret.target_node,
                        level=self._config.interpret.level,
                        amp=self._config.computational.amp,
                    )
                    for i in range(len(data["participant_id"])):
                        mode_id = data[f"{self.maps_manager.mode}_id"][i]
                        cum_maps[mode_id] += map_pt[i]
                        if self._config.interpret.save_individual:
                            single_path = (
                                results_path
                                / f"{data['participant_id'][i]}_{data['session_id'][i]}_{self.maps_manager.mode}-{data[f'{self.maps_manager.mode}_id'][i]}_map.pt"
                            )
                            torch.save(map_pt[i], single_path)
                            if self._config.maps_manager.save_nifti:
                                import nibabel as nib
                                from numpy import eye

                                single_nifti_path = (
                                    results_path
                                    / f"{data['participant_id'][i]}_{data['session_id'][i]}_{self.maps_manager.mode}-{data[f'{self.maps_manager.mode}_id'][i]}_map.nii.gz"
                                )
                                output_nii = nib.nifti1.Nifti1Image(
                                    map_pt[i].numpy(), eye(4)
                                )
                                nib.loadsave.save(output_nii, single_nifti_path)
                for i, mode_map in enumerate(cum_maps):
                    mode_map /= len(data_test)
                    torch.save(
                        mode_map,
                        results_path / f"mean_{self.maps_manager.mode}-{i}_map.pt",
                    )
                    if self._config.maps_manager.save_nifti:
                        import nibabel as nib
                        from numpy import eye

                        output_nii = nib.nifti1.Nifti1Image(mode_map.numpy(), eye(4))
                        nib.loadsave.save(
                            output_nii,
                            results_path
                            / f"mean_{self.maps_manager.mode}-{i}_map.nii.gz",
                        )

    def _check_data_group(
        self,
        df: Optional[pd.DataFrame] = None,
    ):
        """Check if a data group is already available if other arguments are None.
        Else creates a new data_group.

        Parameters
        ----------

        Raises
        ------
        MAPSError
            when trying to overwrite train or validation data groups
        ClinicaDLArgumentError
            when caps_directory or df are given but data group already exists
        ClinicaDLArgumentError
            when caps_directory or df are not given and data group does not exist

        """
        group_dir = (
            self.maps_manager.maps_path
            / "groups"
            / self._config.maps_manager.data_group
        )
        logger.debug(f"Group path {group_dir}")
        print(f"group_dir: {group_dir}")
        if group_dir.is_dir():  # Data group already exists
            print("is dir")
            if self._config.maps_manager.overwrite:
                if self._config.maps_manager.data_group in ["train", "validation"]:
                    raise MAPSError("Cannot overwrite train or validation data group.")
                else:
                    if not self._config.split.split:
                        self._config.split.split = self.maps_manager.find_splits()
                    assert self._config.split
                    for split in self._config.split.split:
                        selection_metrics = find_selection_metrics(
                            self.maps_manager.maps_path,
                            split,
                        )
                        for selection in selection_metrics:
                            results_path = (
                                self.maps_manager.maps_path
                                / f"split-{split}"
                                / f"best-{selection}"
                                / self._config.maps_manager.data_group
                            )
                            if results_path.is_dir():
                                shutil.rmtree(results_path)
            elif df is not None or (
                self._config.data.caps_directory is not None
                and self._config.data.caps_directory != Path("")
            ):
                raise ClinicaDLArgumentError(
                    f"Data group {self._config.maps_manager.data_group} is already defined. "
                    f"Please do not give any caps_directory, tsv_path or multi_cohort to use it. "
                    f"To erase {self._config.maps_manager.data_group} please set overwrite to True."
                )

        elif not group_dir.is_dir() and (
            self._config.data.caps_directory is None or df is None
        ):  # Data group does not exist yet / was overwritten + missing data
            raise ClinicaDLArgumentError(
                f"The data group {self._config.maps_manager.data_group} does not already exist. "
                f"Please specify a caps_directory and a tsv_path to create this data group."
            )
        elif (
            not group_dir.is_dir()
        ):  # Data group does not exist yet / was overwritten + all data is provided
            if self._config.validation.skip_leak_check:
                logger.info("Skipping data leakage check")
            else:
                self._check_leakage(self._config.maps_manager.data_group, df)
            self._write_data_group(
                self._config.maps_manager.data_group,
                df,
                self._config.data.caps_directory,
                self._config.data.multi_cohort,
                label=self._config.data.label,
            )

    def get_group_info(
        self, data_group: str, split: int = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Gets information from corresponding data group
        (list of participant_id / session_id + configuration parameters).
        split is only needed if data_group is train or validation.

        Parameters
        ----------
        data_group : str
            _description_
        split : int (optional, default=None)
            _description_

        Returns
        -------
        Tuple[pd.DataFrame, Dict[str, Any]]
            _description_

        Raises
        ------
        MAPSError
            _description_
        MAPSError
            _description_
        MAPSError
            _description_
        """
        group_path = self.maps_manager.maps_path / "groups" / data_group
        if not group_path.is_dir():
            raise MAPSError(
                f"Data group {data_group} is not defined. "
                f"Please run a prediction to create this data group."
            )
        if data_group in ["train", "validation"]:
            if split is None:
                raise MAPSError(
                    "Information on train or validation data can only be "
                    "loaded if a split number is given"
                )
            elif not (group_path / f"split-{split}").is_dir():
                raise MAPSError(
                    f"Split {split} is not available for data group {data_group}."
                )
            else:
                group_path = group_path / f"split-{split}"

        df = pd.read_csv(group_path / "data.tsv", sep="\t")
        json_path = group_path / "maps.json"
        from clinicadl.utils.iotools.utils import path_decoder

        with json_path.open(mode="r") as f:
            parameters = json.load(f, object_hook=path_decoder)
        return df, parameters

    def _check_leakage(self, data_group: str, test_df: pd.DataFrame):
        """Checks that no intersection exist between the participants used for training and those used for testing.

        Parameters
        ----------
        data_group : str
            name of the data group
        test_df : pd.DataFrame
            Table of participant_id / session_id of the data group

        Raises
        ------
        ClinicaDLDataLeakageError
            if data_group not in ["train", "validation"] and there is an intersection
            between the participant IDs in test_df and the ones used for training.
        """
        if data_group not in ["train", "validation"]:
            train_path = self.maps_manager.maps_path / "groups" / "train+validation.tsv"
            train_df = pd.read_csv(train_path, sep="\t")
            participants_train = set(train_df.participant_id.values)
            participants_test = set(test_df.participant_id.values)
            intersection = participants_test & participants_train

            if len(intersection) > 0:
                raise ClinicaDLDataLeakageError(
                    "Your evaluation set contains participants who were already seen during "
                    "the training step. The list of common participants is the following: "
                    f"{intersection}."
                )

    def _write_data_group(
        self,
        data_group,
        df,
        caps_directory: Path = None,
        multi_cohort: bool = None,
        label=None,
    ):
        """Check that a data_group is not already written and writes the characteristics of the data group
        (TSV file with a list of participant / session + JSON file containing the CAPS and the preprocessing).

        Parameters
        ----------
        data_group : _type_
            name whose presence is checked.
        df : _type_
            DataFrame containing the participant_id and session_id (and label if use_labels is True)
        caps_directory : Path (optional, default=None)
            caps_directory if different from the training caps_directory,
        multi_cohort : bool (optional, default=None)
            multi_cohort used if different from the training multi_cohort.
        label : _type_ (optional, default=None)
            _description_
        """
        group_path = self.maps_manager.maps_path / "groups" / data_group
        group_path.mkdir(parents=True)

        columns = ["participant_id", "session_id", "cohort"]
        if self._config.data.label in df.columns.values:
            columns += [self._config.data.label]
        if label is not None and label in df.columns.values:
            columns += [label]

        df.to_csv(group_path / "data.tsv", sep="\t", columns=columns, index=False)
        self.maps_manager.write_parameters(
            group_path,
            {
                "caps_directory": (
                    caps_directory
                    if caps_directory is not None
                    else self._config.caps_directory
                ),
                "multi_cohort": (
                    multi_cohort
                    if multi_cohort is not None
                    else self._config.multi_cohort
                ),
            },
        )

    # this function is never used ???

    def get_interpretation(
        self,
        data_group: str,
        name: str,
        split: int = 0,
        selection_metric: Optional[str] = None,
        verbose: bool = True,
        participant_id: Optional[str] = None,
        session_id: Optional[str] = None,
        mode_id: int = 0,
    ) -> torch.Tensor:
        """
        Get the individual interpretation maps for one session if participant_id and session_id are filled.
        Else load the mean interpretation map.

        Args:
            data_group (str): Name of the data group used for the interpretation task.
            name (str): name of the interpretation task.
            split (int): Index of the split used for training.
            selection_metric (str): Metric used for best weights selection.
            verbose (bool): if True will print associated prediction.log.
            participant_id (str): ID of the participant (if not given load mean map).
            session_id (str): ID of the session (if not give load the mean map).
            mode_id (int): Index of the mode used.
        Returns:
            (torch.Tensor): Tensor of the interpretability map.
        """

        selection_metric = check_selection_metric(
            self.maps_manager.maps_path,
            split,
            selection_metric,
        )
        if verbose:
            self.maps_manager._print_description_log(
                data_group, split, selection_metric
            )
        map_dir = (
            self.maps_manager.maps_path
            / f"split-{split}"
            / f"best-{selection_metric}"
            / data_group
            / f"interpret-{name}"
        )
        if not map_dir.is_dir():
            raise MAPSError(
                f"No prediction corresponding to data group {data_group} and "
                f"interpretation {name} was found."
            )
        if participant_id is None and session_id is None:
            map_pt = torch.load(
                map_dir / f"mean_{self.maps_manager.mode}-{mode_id}_map.pt",
                weights_only=True,
            )
        elif participant_id is None or session_id is None:
            raise ValueError(
                "To load the mean interpretation map, "
                "please do not give any participant_id or session_id.\n "
                "Else specify both parameters"
            )
        else:
            map_pt = torch.load(
                map_dir
                / f"{participant_id}_{session_id}_{self.maps_manager.mode}-{mode_id}_map.pt",
                weights_only=True,
            )
        return map_pt

    def test(
        self,
        mode: str,
        metrics_module: MetricModule,
        n_classes: int,
        network_task,
        model: Network,
        dataloader: DataLoader,
        criterion: _Loss,
        use_labels: bool = True,
        amp: bool = False,
        report_ci=False,
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Computes the predictions and evaluation metrics.

        Parameters
        ----------
        model: Network
            The model trained.
        dataloader: DataLoader
            Wrapper of a CapsDataset.
        criterion:  _Loss
            Function to calculate the loss.
        use_labels: bool
            If True the true_label will be written in output DataFrame
            and metrics dict will be created.
        amp: bool
            If True, enables Pytorch's automatic mixed precision.

        Returns
        -------
            the results and metrics on the image level.
        """
        model.eval()
        dataloader.dataset.eval()

        results_df = pd.DataFrame(columns=columns(network_task, mode, n_classes))
        total_loss = {}
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                # initialize the loss list to save the loss components
                with autocast("cuda", enabled=amp):
                    outputs, loss_dict = model(data, criterion, use_labels=use_labels)

                if i == 0:
                    for loss_component in loss_dict.keys():
                        total_loss[loss_component] = 0
                for loss_component in total_loss.keys():
                    total_loss[loss_component] += loss_dict[loss_component].float()

                # Generate detailed DataFrame
                for idx in range(len(data["participant_id"])):
                    row = generate_test_row(
                        network_task,
                        mode,
                        metrics_module,
                        n_classes,
                        idx,
                        data,
                        outputs.float(),
                    )
                    row_df = pd.DataFrame(
                        row, columns=columns(network_task, mode, n_classes)
                    )
                    results_df = pd.concat([results_df, row_df])

                del outputs, loss_dict
        dataframes = [None] * dist.get_world_size()
        dist.gather_object(
            results_df, dataframes if dist.get_rank() == 0 else None, dst=0
        )
        if dist.get_rank() == 0:
            results_df = pd.concat(dataframes)
        del dataframes
        results_df.reset_index(inplace=True, drop=True)

        if not use_labels:
            metrics_dict = None
        else:
            metrics_dict = compute_metrics(
                network_task, results_df, metrics_module, report_ci=report_ci
            )
            for loss_component in total_loss.keys():
                dist.reduce(total_loss[loss_component], dst=0)
                loss_value = total_loss[loss_component].item() / cluster.world_size

                if report_ci:
                    metrics_dict["Metric_names"].append(loss_component)
                    metrics_dict["Metric_values"].append(loss_value)
                    metrics_dict["Lower_CI"].append("N/A")
                    metrics_dict["Upper_CI"].append("N/A")
                    metrics_dict["SE"].append("N/A")

                else:
                    metrics_dict[loss_component] = loss_value

        torch.cuda.empty_cache()

        return results_df, metrics_dict

    def test_da(
        self,
        mode: str,
        metrics_module: MetricModule,
        n_classes: int,
        network_task: Union[str, Task],
        model: Network,
        dataloader: DataLoader,
        criterion: _Loss,
        alpha: float = 0,
        use_labels: bool = True,
        target: bool = True,
        report_ci=False,
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Computes the predictions and evaluation metrics.

        Args:
            model: the model trained.
            dataloader: wrapper of a CapsDataset.
            criterion: function to calculate the loss.
            use_labels: If True the true_label will be written in output DataFrame
                and metrics dict will be created.
        Returns:
            the results and metrics on the image level.
        """
        model.eval()
        dataloader.dataset.eval()
        results_df = pd.DataFrame(columns=columns(network_task, mode, n_classes))
        total_loss = 0
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                outputs, loss_dict = model.compute_outputs_and_loss_test(
                    data, criterion, alpha, target
                )
                total_loss += loss_dict["loss"].item()

                # Generate detailed DataFrame
                for idx in range(len(data["participant_id"])):
                    row = generate_test_row(
                        network_task,
                        mode,
                        metrics_module,
                        n_classes,
                        idx,
                        data,
                        outputs,
                    )
                    row_df = pd.DataFrame(
                        row, columns=columns(network_task, mode, n_classes)
                    )
                    results_df = pd.concat([results_df, row_df])

                del outputs, loss_dict
            results_df.reset_index(inplace=True, drop=True)

        if not use_labels:
            metrics_dict = None
        else:
            metrics_dict = compute_metrics(
                network_task, results_df, metrics_module, report_ci=report_ci
            )
            if report_ci:
                metrics_dict["Metric_names"].append("loss")
                metrics_dict["Metric_values"].append(total_loss)
                metrics_dict["Lower_CI"].append("N/A")
                metrics_dict["Upper_CI"].append("N/A")
                metrics_dict["SE"].append("N/A")

            else:
                metrics_dict["loss"] = total_loss

        torch.cuda.empty_cache()

        return results_df, metrics_dict

    def _test_loader(
        self,
        maps_manager: MapsManager,
        dataloader,
        criterion,
        data_group: str,
        split: int,
        selection_metrics,
        use_labels=True,
        gpu=None,
        amp=False,
        network=None,
        report_ci=True,
    ):
        """
        Launches the testing task on a dataset wrapped by a DataLoader and writes prediction TSV files.

        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader wrapping the test CapsDataset.
            criterion (torch.nn.modules.loss._Loss): optimization criterion used during training.
            data_group (str): name of the data group used for the testing task.
            split (int): Index of the split used to train the model tested.
            selection_metrics (list[str]): List of metrics used to select the best models which are tested.
            use_labels (bool): If True, the labels must exist in test meta-data and metrics are computed.
            gpu (bool): If given, a new value for the device of the model will be computed.
            amp (bool): If enabled, uses Automatic Mixed Precision (requires GPU usage).
            network (int): Index of the network tested (only used in multi-network setting).
        """
        for selection_metric in selection_metrics:
            if cluster.master:
                log_dir = (
                    maps_manager.maps_path
                    / f"split-{split}"
                    / f"best-{selection_metric}"
                    / data_group
                )
                maps_manager.write_description_log(
                    log_dir,
                    data_group,
                    dataloader.dataset.config.data.caps_dict,
                    dataloader.dataset.config.data.data_df,
                )

            # load the best trained model during the training
            model, _ = maps_manager._init_model(
                transfer_path=maps_manager.maps_path,
                split=split,
                transfer_selection=selection_metric,
                gpu=gpu,
                network=network,
            )
            model = DDP(
                model,
                fsdp=maps_manager.fully_sharded_data_parallel,
                amp=maps_manager.amp,
            )

            prediction_df, metrics = self.test(
                mode=maps_manager.mode,
                metrics_module=maps_manager.metrics_module,
                n_classes=maps_manager.n_classes,
                network_task=maps_manager.network_task,
                model=model,
                dataloader=dataloader,
                criterion=criterion,
                use_labels=use_labels,
                amp=amp,
                report_ci=report_ci,
            )
            if use_labels:
                if network is not None:
                    metrics[f"{maps_manager.mode}_id"] = network

                loss_to_log = (
                    metrics["Metric_values"][-1] if report_ci else metrics["loss"]
                )

                logger.info(
                    f"{maps_manager.mode} level {data_group} loss is {loss_to_log} for model selected on {selection_metric}"
                )

            if cluster.master:
                # Replace here
                print("before saving")
                maps_manager._mode_level_to_tsv(
                    prediction_df,
                    metrics,
                    split,
                    selection_metric,
                    data_group=data_group,
                )

    def _test_loader_ssda(
        self,
        maps_manager: MapsManager,
        dataloader,
        criterion,
        alpha,
        data_group,
        split,
        selection_metrics,
        use_labels=True,
        gpu=None,
        network=None,
        target=False,
        report_ci=True,
    ):
        """
        Launches the testing task on a dataset wrapped by a DataLoader and writes prediction TSV files.

        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader wrapping the test CapsDataset.
            criterion (torch.nn.modules.loss._Loss): optimization criterion used during training.
            data_group (str): name of the data group used for the testing task.
            split (int): Index of the split used to train the model tested.
            selection_metrics (list[str]): List of metrics used to select the best models which are tested.
            use_labels (bool): If True, the labels must exist in test meta-data and metrics are computed.
            gpu (bool): If given, a new value for the device of the model will be computed.
            network (int): Index of the network tested (only used in multi-network setting).
        """
        for selection_metric in selection_metrics:
            log_dir = (
                maps_manager.maps_path
                / f"split-{split}"
                / f"best-{selection_metric}"
                / data_group
            )
            maps_manager.write_description_log(
                log_dir,
                data_group,
                dataloader.dataset.caps_dict,
                dataloader.dataset.df,
            )

            # load the best trained model during the training
            model, _ = maps_manager._init_model(
                transfer_path=maps_manager.maps_path,
                split=split,
                transfer_selection=selection_metric,
                gpu=gpu,
                network=network,
            )
            prediction_df, metrics = self.test_da(
                network_task=maps_manager.network_task,
                model=model,
                dataloader=dataloader,
                criterion=criterion,
                target=target,
                report_ci=report_ci,
                mode=maps_manager.mode,
                metrics_module=maps_manager.metrics_module,
                n_classes=maps_manager.n_classes,
            )
            if use_labels:
                if network is not None:
                    metrics[f"{maps_manager.mode}_id"] = network

                if report_ci:
                    loss_to_log = metrics["Metric_values"][-1]
                else:
                    loss_to_log = metrics["loss"]

                logger.info(
                    f"{maps_manager.mode} level {data_group} loss is {loss_to_log} for model selected on {selection_metric}"
                )

            # Replace here
            maps_manager._mode_level_to_tsv(
                prediction_df, metrics, split, selection_metric, data_group=data_group
            )

    @torch.no_grad()
    def _compute_output_tensors(
        self,
        maps_manager: MapsManager,
        dataset,
        data_group,
        split,
        selection_metrics,
        nb_images=None,
        gpu=None,
        network=None,
    ):
        """
        Compute the output tensors and saves them in the MAPS.

        Args:
            dataset (clinicadl.caps_dataset.data.CapsDataset): wrapper of the data set.
            data_group (str): name of the data group used for the task.
            split (int): split number.
            selection_metrics (list[str]): metrics used for model selection.
            nb_images (int): number of full images to write. Default computes the outputs of the whole data set.
            gpu (bool): If given, a new value for the device of the model will be computed.
            network (int): Index of the network tested (only used in multi-network setting).
        """
        for selection_metric in selection_metrics:
            # load the best trained model during the training
            model, _ = maps_manager._init_model(
                transfer_path=maps_manager.maps_path,
                split=split,
                transfer_selection=selection_metric,
                gpu=gpu,
                network=network,
                nb_unfrozen_layer=maps_manager.nb_unfrozen_layer,
            )
            model = DDP(
                model,
                fsdp=maps_manager.fully_sharded_data_parallel,
                amp=maps_manager.amp,
            )
            model.eval()

            tensor_path = (
                maps_manager.maps_path
                / f"split-{split}"
                / f"best-{selection_metric}"
                / data_group
                / "tensors"
            )
            if cluster.master:
                tensor_path.mkdir(parents=True, exist_ok=True)
            dist.barrier()

            if nb_images is None:  # Compute outputs for the whole data set
                nb_modes = len(dataset)
            else:
                nb_modes = nb_images * dataset.elem_per_image

            for i in [
                *range(cluster.rank, nb_modes, cluster.world_size),
                *range(int(nb_modes % cluster.world_size <= cluster.rank)),
            ]:
                data = dataset[i]
                image = data["image"]
                x = image.unsqueeze(0).to(model.device)
                with autocast("cuda", enabled=maps_manager.std_amp):
                    output = model(x)
                output = output.squeeze(0).cpu().float()
                participant_id = data["participant_id"]
                session_id = data["session_id"]
                mode_id = data[f"{maps_manager.mode}_id"]
                input_filename = f"{participant_id}_{session_id}_{maps_manager.mode}-{mode_id}_input.pt"
                output_filename = f"{participant_id}_{session_id}_{maps_manager.mode}-{mode_id}_output.pt"
                torch.save(image, tensor_path / input_filename)
                torch.save(output, tensor_path / output_filename)
                logger.debug(f"File saved at {[input_filename, output_filename]}")

    def _ensemble_prediction(
        self,
        maps_manager: MapsManager,
        data_group,
        split,
        selection_metrics,
        use_labels=True,
        skip_leak_check=False,
    ):
        """Computes the results on the image-level."""

        if not selection_metrics:
            selection_metrics = find_selection_metrics(maps_manager.maps_path, split)

        for selection_metric in selection_metrics:
            #####################
            # Soft voting
            if maps_manager.num_networks > 1 and not skip_leak_check:
                maps_manager._ensemble_to_tsv(
                    split,
                    selection=selection_metric,
                    data_group=data_group,
                    use_labels=use_labels,
                )
            elif maps_manager.mode != "image" and not skip_leak_check:
                maps_manager._mode_to_image_tsv(
                    split,
                    selection=selection_metric,
                    data_group=data_group,
                    use_labels=use_labels,
                )
