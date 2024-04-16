import json
import shutil
import subprocess
from contextlib import nullcontext
from datetime import datetime
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from clinicadl.utils.callbacks.callbacks import Callback, CallbacksHandler
from clinicadl.utils.caps_dataset.data import (
    get_transforms,
    load_data_test,
    return_dataset,
)
from clinicadl.utils.cmdline_utils import check_gpu
from clinicadl.utils.early_stopping import EarlyStopping
from clinicadl.utils.exceptions import (
    ClinicaDLArgumentError,
    ClinicaDLConfigurationError,
    ClinicaDLDataLeakageError,
    MAPSError,
)
from clinicadl.utils.maps_manager.ddp import DDP, cluster, init_ddp
from clinicadl.utils.maps_manager.logwriter import LogWriter
from clinicadl.utils.maps_manager.maps_manager import MapsManager
from clinicadl.utils.maps_manager.maps_manager_utils import (
    add_default_values,
    read_json,
)
from clinicadl.utils.metric_module import RetainBest
from clinicadl.utils.network.network import Network
from clinicadl.utils.predict_manager.predict_config import PredictConfig
from clinicadl.utils.preprocessing import path_decoder, path_encoder
from clinicadl.utils.seed import get_seed, pl_worker_init_function, seed_everything

logger = getLogger("clinicadl.maps_manager")
level_list: List[str] = ["warning", "info", "debug"]


class PredictManager:
    def __init__(self, maps_manager: MapsManager):
        self.maps_manager = maps_manager
        # self.predict_config = PredictConfig()

    def predict(
        self,
        data_group: str,
        caps_directory: Path = None,
        tsv_path: Path = None,
        split_list: List[int] = None,
        selection_metrics: List[str] = None,
        multi_cohort: bool = False,
        diagnoses: List[str] = (),
        use_labels: bool = True,
        batch_size: int = None,
        n_proc: int = None,
        gpu: bool = None,
        amp: bool = False,
        overwrite: bool = False,
        label: str = None,
        label_code: Optional[Dict[str, int]] = "default",
        save_tensor: bool = False,
        save_nifti: bool = False,
        save_latent_tensor: bool = False,
        skip_leak_check: bool = False,
    ):
        """
        Performs the prediction task on a subset of caps_directory defined in a TSV file.

        Args:
            data_group: name of the data group tested.
            caps_directory: path to the CAPS folder. For more information please refer to
                [clinica documentation](https://aramislab.paris.inria.fr/clinica/docs/public/latest/CAPS/Introduction/).
                Default will load the value of an existing data group
            tsv_path: path to a TSV file containing the list of participants and sessions to test.
                Default will load the DataFrame of an existing data group
            split_list: list of splits to test. Default perform prediction on all splits available.
            selection_metrics (list[str]): list of selection metrics to test.
                Default performs the prediction on all selection metrics available.
            multi_cohort: If True considers that tsv_path is the path to a multi-cohort TSV.
            diagnoses: List of diagnoses to load if tsv_path is a split_directory.
                Default uses the same as in training step.
            use_labels: If True, the labels must exist in test meta-data and metrics are computed.
            batch_size: If given, sets the value of batch_size, else use the same as in training step.
            n_proc: If given, sets the value of num_workers, else use the same as in training step.
            gpu: If given, a new value for the device of the model will be computed.
            amp: If enabled, uses Automatic Mixed Precision (requires GPU usage).
            overwrite: If True erase the occurrences of data_group.
            label: Target label used for training (if network_task in [`regression`, `classification`]).
            label_code: dictionary linking the target values to a node number.
        """
        if not split_list:
            split_list = self.maps_manager._find_splits()
        logger.debug(f"List of splits {split_list}")

        _, all_transforms = get_transforms(
            normalize=self.maps_manager.normalize,
            data_augmentation=self.maps_manager.data_augmentation,
            size_reduction=self.maps_manager.size_reduction,
            size_reduction_factor=self.maps_manager.size_reduction_factor,
        )

        group_df = None
        if tsv_path is not None:
            group_df = load_data_test(
                tsv_path,
                diagnoses if len(diagnoses) != 0 else self.maps_manager.diagnoses,
                multi_cohort=multi_cohort,
            )
        criterion = self.maps_manager.task_manager.get_criterion(self.maps_manager.loss)
        self._check_data_group(
            data_group,
            caps_directory,
            group_df,
            multi_cohort,
            overwrite,
            label=label,
            split_list=split_list,
            skip_leak_check=skip_leak_check,
        )
        for split in split_list:
            logger.info(f"Prediction of split {split}")
            group_df, group_parameters = self.get_group_info(data_group, split)
            # Find label code if not given
            if (
                label is not None
                and label != self.maps_manager.label
                and label_code == "default"
            ):
                self.maps_manager.task_manager.generate_label_code(group_df, label)

            # Erase previous TSV files on master process
            if not selection_metrics:
                split_selection_metrics = self.maps_manager._find_selection_metrics(
                    split
                )
            else:
                split_selection_metrics = selection_metrics
            for selection in split_selection_metrics:
                tsv_dir = (
                    self.maps_manager.maps_path
                    / f"{self.maps_manager.split_name}-{split}"
                    / f"best-{selection}"
                    / data_group
                )

                tsv_pattern = f"{data_group}*.tsv"

                for tsv_file in tsv_dir.glob(tsv_pattern):
                    tsv_file.unlink()

            if self.maps_manager.multi_network:
                self._predict_multi(
                    group_parameters,
                    group_df,
                    all_transforms,
                    use_labels,
                    label,
                    label_code,
                    batch_size,
                    n_proc,
                    criterion,
                    data_group,
                    split,
                    split_selection_metrics,
                    gpu,
                    amp,
                    save_tensor,
                    save_latent_tensor,
                    save_nifti,
                    selection_metrics,
                )

            else:
                self._predict_single(
                    group_parameters,
                    group_df,
                    all_transforms,
                    use_labels,
                    label,
                    label_code,
                    batch_size,
                    n_proc,
                    criterion,
                    data_group,
                    split,
                    split_selection_metrics,
                    gpu,
                    amp,
                    save_tensor,
                    save_latent_tensor,
                    save_nifti,
                    selection_metrics,
                )

            if cluster.master:
                self.maps_manager._ensemble_prediction(
                    data_group, split, selection_metrics, use_labels, skip_leak_check
                )

    def _predict_multi(
        self,
        group_parameters,
        group_df,
        all_transforms,
        use_labels,
        label,
        label_code,
        batch_size,
        n_proc,
        criterion,
        data_group,
        split,
        split_selection_metrics,
        gpu,
        amp,
        save_tensor,
        save_latent_tensor,
        save_nifti,
        selection_metrics,
    ):
        for network in range(self.maps_manager.num_networks):
            data_test = return_dataset(
                group_parameters["caps_directory"],
                group_df,
                self.maps_manager.preprocessing_dict,
                all_transformations=all_transforms,
                multi_cohort=group_parameters["multi_cohort"],
                label_presence=use_labels,
                label=self.maps_manager.label if label is None else label,
                label_code=(
                    self.maps_manager.label_code
                    if label_code == "default"
                    else label_code
                ),
                cnn_index=network,
            )
            test_loader = DataLoader(
                data_test,
                batch_size=(
                    batch_size
                    if batch_size is not None
                    else self.maps_manager.batch_size
                ),
                shuffle=False,
                sampler=DistributedSampler(
                    data_test,
                    num_replicas=cluster.world_size,
                    rank=cluster.rank,
                    shuffle=False,
                ),
                num_workers=n_proc if n_proc is not None else self.maps_manager.n_proc,
            )
            self.maps_manager._test_loader(
                test_loader,
                criterion,
                data_group,
                split,
                split_selection_metrics,
                use_labels=use_labels,
                gpu=gpu,
                amp=amp,
                network=network,
            )
            if save_tensor:
                logger.debug("Saving tensors")
                self.maps_manager._compute_output_tensors(
                    data_test,
                    data_group,
                    split,
                    selection_metrics,
                    gpu=gpu,
                    network=network,
                )
            if save_nifti:
                self._compute_output_nifti(
                    data_test,
                    data_group,
                    split,
                    selection_metrics,
                    gpu=gpu,
                    network=network,
                )
            if save_latent_tensor:
                self._compute_latent_tensors(
                    data_test,
                    data_group,
                    split,
                    selection_metrics,
                    gpu=gpu,
                    network=network,
                )

    def _predict_single(
        self,
        group_parameters,
        group_df,
        all_transforms,
        use_labels,
        label,
        label_code,
        batch_size,
        n_proc,
        criterion,
        data_group,
        split,
        split_selection_metrics,
        gpu,
        amp,
        save_tensor,
        save_latent_tensor,
        save_nifti,
        selection_metrics,
    ):
        data_test = return_dataset(
            group_parameters["caps_directory"],
            group_df,
            self.maps_manager.preprocessing_dict,
            all_transformations=all_transforms,
            multi_cohort=group_parameters["multi_cohort"],
            label_presence=use_labels,
            label=self.maps_manager.label if label is None else label,
            label_code=(
                self.maps_manager.label_code if label_code == "default" else label_code
            ),
        )

        test_loader = DataLoader(
            data_test,
            batch_size=(
                batch_size if batch_size is not None else self.maps_manager.batch_size
            ),
            shuffle=False,
            sampler=DistributedSampler(
                data_test,
                num_replicas=cluster.world_size,
                rank=cluster.rank,
                shuffle=False,
            ),
            num_workers=n_proc if n_proc is not None else self.maps_manager.n_proc,
        )
        self.maps_manager._test_loader(
            test_loader,
            criterion,
            data_group,
            split,
            split_selection_metrics,
            use_labels=use_labels,
            gpu=gpu,
            amp=amp,
        )
        if save_tensor:
            logger.debug("Saving tensors")
            self.maps_manager._compute_output_tensors(
                data_test,
                data_group,
                split,
                selection_metrics,
                gpu=gpu,
            )
        if save_nifti:
            self._compute_output_nifti(
                data_test,
                data_group,
                split,
                selection_metrics,
                gpu=gpu,
            )
        if save_latent_tensor:
            self._compute_latent_tensors(
                data_test,
                data_group,
                split,
                selection_metrics,
                gpu=gpu,
            )

    def _compute_latent_tensors(
        self,
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
            dataset (clinicadl.utils.caps_dataset.data.CapsDataset): wrapper of the data set.
            data_group (str): name of the data group used for the task.
            split (int): split number.
            selection_metrics (list[str]): metrics used for model selection.
            nb_images (int): number of full images to write. Default computes the outputs of the whole data set.
            gpu (bool): If given, a new value for the device of the model will be computed.
            network (int): Index of the network tested (only used in multi-network setting).
        """
        for selection_metric in selection_metrics:
            # load the best trained model during the training
            model, _ = self.maps_manager._init_model(
                transfer_path=self.maps_manager.maps_path,
                split=split,
                transfer_selection=selection_metric,
                gpu=gpu,
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
                / f"{self.maps_manager.split_name}-{split}"
                / f"best-{selection_metric}"
                / data_group
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
                with autocast(enabled=self.maps_manager.std_amp):
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
        data_group,
        split,
        selection_metrics,
        gpu=None,
        network=None,
    ):
        """
        Computes the output nifti images and saves them in the MAPS.

        Args:
            dataset (clinicadl.utils.caps_dataset.data.CapsDataset): wrapper of the data set.
            data_group (str): name of the data group used for the task.
            split (int): split number.
            selection_metrics (list[str]): metrics used for model selection.
            gpu (bool): If given, a new value for the device of the model will be computed.
            network (int): Index of the network tested (only used in multi-network setting).
        # Raise an error if mode is not image
        """
        import nibabel as nib
        from numpy import eye

        for selection_metric in selection_metrics:
            # load the best trained model during the training
            model, _ = self.maps_manager._init_model(
                transfer_path=self.maps_manager.maps_path,
                split=split,
                transfer_selection=selection_metric,
                gpu=gpu,
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
                / f"{self.maps_manager.split_name}-{split}"
                / f"best-{selection_metric}"
                / data_group
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
                with autocast(enabled=self.maps_manager.std_amp):
                    output = model(x)
                output = output.squeeze(0).detach().cpu().float()
                # Convert tensor to nifti image with appropriate affine
                input_nii = nib.Nifti1Image(image[0].detach().cpu().numpy(), eye(4))
                output_nii = nib.Nifti1Image(output[0].numpy(), eye(4))
                # Create file name according to participant and session id
                participant_id = data["participant_id"]
                session_id = data["session_id"]
                input_filename = f"{participant_id}_{session_id}_image_input.nii.gz"
                output_filename = f"{participant_id}_{session_id}_image_output.nii.gz"
                nib.save(input_nii, nifti_path / input_filename)
                nib.save(output_nii, nifti_path / output_filename)

    def interpret(
        self,
        data_group,
        name,
        method,
        caps_directory: Path = None,
        tsv_path: Path = None,
        split_list=None,
        selection_metrics=None,
        multi_cohort=False,
        diagnoses=(),
        target_node=0,
        save_individual=False,
        batch_size=None,
        n_proc=None,
        gpu=None,
        amp=False,
        overwrite=False,
        overwrite_name=False,
        level=None,
        save_nifti=False,
    ):
        """
        Performs the interpretation task on a subset of caps_directory defined in a TSV file.
        The mean interpretation is always saved, to save the individual interpretations set save_individual to True.

        Parameters
        ----------
        data_group: str
            Name of the data group interpreted.
        name: str
            Name of the interpretation procedure.
        method: str
            Method used for extraction (ex: gradients, grad-cam...).
        caps_directory: str (Path)
            Path to the CAPS folder. For more information please refer to
            [clinica documentation](https://aramislab.paris.inria.fr/clinica/docs/public/latest/CAPS/Introduction/).
            Default will load the value of an existing data group.
        tsv_path: str (Path)
            Path to a TSV file containing the list of participants and sessions to test.
            Default will load the DataFrame of an existing data group.
        split_list: list of int
            List of splits to interpret. Default perform interpretation on all splits available.
        selection_metrics: list of str
            List of selection metrics to interpret.
            Default performs the interpretation on all selection metrics available.
        multi_cohort: bool
            If True considers that tsv_path is the path to a multi-cohort TSV.
        diagnoses: list of str
            List of diagnoses to load if tsv_path is a split_directory.
            Default uses the same as in training step.
        target_node: int
            Node from which the interpretation is computed.
        save_individual: bool
            If True saves the individual map of each participant / session couple.
        batch_size: int
            If given, sets the value of batch_size, else use the same as in training step.
        n_proc: int
            If given, sets the value of num_workers, else use the same as in training step.
        gpu: bool
            If given, a new value for the device of the model will be computed.
        amp: bool
            If enabled, uses Automatic Mixed Precision (requires GPU usage).
        overwrite: bool
            If True erase the occurrences of data_group.
        overwrite_name: bool
            If True erase the occurrences of name.
        level: int
            Layer number in the convolutional part after which the feature map is chosen.
        save_nifi : bool
            If True, save the interpretation map in nifti format.
        """

        from clinicadl.interpret.gradients import method_dict

        if method not in method_dict.keys():
            raise NotImplementedError(
                f"Interpretation method {method} is not implemented. "
                f"Please choose in {method_dict.keys()}"
            )

        if not split_list:
            split_list = self.maps_manager._find_splits()
        logger.debug(f"List of splits {split_list}")

        if self.maps_manager.multi_network:
            raise NotImplementedError(
                "The interpretation of multi-network framework is not implemented."
            )

        _, all_transforms = get_transforms(
            normalize=self.maps_manager.normalize,
            data_augmentation=self.maps_manager.data_augmentation,
            size_reduction=self.maps_manager.size_reduction,
            size_reduction_factor=self.maps_manager.size_reduction_factor,
        )

        group_df = None
        if tsv_path is not None:
            group_df = load_data_test(
                tsv_path,
                diagnoses if len(diagnoses) != 0 else self.maps_manager.diagnoses,
                multi_cohort=multi_cohort,
            )
        self._check_data_group(
            data_group, caps_directory, group_df, multi_cohort, overwrite
        )

        for split in split_list:
            logger.info(f"Interpretation of split {split}")
            df_group, parameters_group = self.get_group_info(data_group, split)

            data_test = return_dataset(
                parameters_group["caps_directory"],
                df_group,
                self.maps_manager.preprocessing_dict,
                all_transformations=all_transforms,
                multi_cohort=parameters_group["multi_cohort"],
                label_presence=False,
                label_code=self.maps_manager.label_code,
                label=self.maps_manager.label,
            )

            test_loader = DataLoader(
                data_test,
                batch_size=batch_size
                if batch_size is not None
                else self.maps_manager.batch_size,
                shuffle=False,
                num_workers=n_proc if n_proc is not None else self.maps_manager.n_proc,
            )

            if not selection_metrics:
                selection_metrics = self.maps_manager._find_selection_metrics(split)

            for selection_metric in selection_metrics:
                logger.info(f"Interpretation of metric {selection_metric}")
                results_path = (
                    self.maps_manager.maps_path
                    / f"{self.maps_manager.split_name}-{split}"
                    / f"best-{selection_metric}"
                    / data_group
                    / f"interpret-{name}"
                )

                if (results_path).is_dir():
                    if overwrite_name:
                        shutil.rmtree(results_path)
                    else:
                        raise MAPSError(
                            f"Interpretation name {name} is already written. "
                            f"Please choose another name or set overwrite_name to True."
                        )
                results_path.mkdir(parents=True)

                model, _ = self.maps_manager._init_model(
                    transfer_path=self.maps_manager.maps_path,
                    split=split,
                    transfer_selection=selection_metric,
                    gpu=gpu,
                )

                interpreter = method_dict[method](model)

                cum_maps = [0] * data_test.elem_per_image
                for data in test_loader:
                    images = data["image"].to(model.device)

                    map_pt = interpreter.generate_gradients(
                        images, target_node, level=level, amp=amp
                    )
                    for i in range(len(data["participant_id"])):
                        mode_id = data[f"{self.maps_manager.mode}_id"][i]
                        cum_maps[mode_id] += map_pt[i]
                        if save_individual:
                            single_path = (
                                results_path
                                / f"{data['participant_id'][i]}_{data['session_id'][i]}_{self.maps_manager.mode}-{data[f'{self.maps_manager.mode}_id'][i]}_map.pt"
                            )
                            torch.save(map_pt[i], single_path)
                            if save_nifti:
                                import nibabel as nib
                                from numpy import eye

                                single_nifti_path = (
                                    results_path
                                    / f"{data['participant_id'][i]}_{data['session_id'][i]}_{self.maps_manager.mode}-{data[f'{self.maps_manager.mode}_id'][i]}_map.nii.gz"
                                )

                                output_nii = nib.Nifti1Image(map_pt[i].numpy(), eye(4))
                                nib.save(output_nii, single_nifti_path)

                for i, mode_map in enumerate(cum_maps):
                    mode_map /= len(data_test)

                    torch.save(
                        mode_map,
                        results_path / f"mean_{self.maps_manager.mode}-{i}_map.pt",
                    )
                    if save_nifti:
                        import nibabel as nib
                        from numpy import eye

                        output_nii = nib.Nifti1Image(mode_map.numpy(), eye(4))
                        nib.save(
                            output_nii,
                            results_path
                            / f"mean_{self.maps_manager.mode}-{i}_map.nii.gz",
                        )

    def _check_data_group(
        self,
        data_group,
        caps_directory=None,
        df=None,
        multi_cohort=False,
        overwrite=False,
        label=None,
        split_list=None,
        skip_leak_check=False,
    ):
        """
        Check if a data group is already available if other arguments are None.
        Else creates a new data_group.

        Args:
            data_group (str): name of the data group
            caps_directory  (str): input CAPS directory
            df (pd.DataFrame): Table of participant_id / session_id of the data group
            multi_cohort (bool): indicates if the input data comes from several CAPS
            overwrite (bool): If True former definition of data group is erased
            label (str): label name if applicable

        Raises:
            MAPSError when trying to overwrite train or validation data groups
            ClinicaDLArgumentError:
                when caps_directory or df are given but data group already exists
                when caps_directory or df are not given and data group does not exist
        """
        group_dir = self.maps_manager.maps_path / "groups" / data_group
        logger.debug(f"Group path {group_dir}")
        if group_dir.is_dir():  # Data group already exists
            if overwrite:
                if data_group in ["train", "validation"]:
                    raise MAPSError("Cannot overwrite train or validation data group.")
                else:
                    if not split_list:
                        split_list = self.maps_manager._find_splits()
                    for split in split_list:
                        selection_metrics = self.maps_manager._find_selection_metrics(
                            split
                        )
                        for selection in selection_metrics:
                            results_path = (
                                self.maps_manager.maps_path
                                / f"{self.maps_manager.split_name}-{split}"
                                / f"best-{selection}"
                                / data_group
                            )
                            if results_path.is_dir():
                                shutil.rmtree(results_path)
            elif df is not None or caps_directory is not None:
                raise ClinicaDLArgumentError(
                    f"Data group {data_group} is already defined. "
                    f"Please do not give any caps_directory, tsv_path or multi_cohort to use it. "
                    f"To erase {data_group} please set overwrite to True."
                )

        elif not group_dir.is_dir() and (
            caps_directory is None or df is None
        ):  # Data group does not exist yet / was overwritten + missing data
            raise ClinicaDLArgumentError(
                f"The data group {data_group} does not already exist. "
                f"Please specify a caps_directory and a tsv_path to create this data group."
            )
        elif (
            not group_dir.is_dir()
        ):  # Data group does not exist yet / was overwritten + all data is provided
            if skip_leak_check:
                logger.info("Skipping data leakage check")
            else:
                self._check_leakage(data_group, df)
            self.maps_manager._write_data_group(
                data_group, df, caps_directory, multi_cohort, label=label
            )

    def get_group_info(
        self, data_group: str, split: int = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Gets information from corresponding data group
        (list of participant_id / session_id + configuration parameters).
        split is only needed if data_group is train or validation.
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
                    f"Information on train or validation data can only be "
                    f"loaded if a split number is given"
                )
            elif not (group_path / f"{self.maps_manager.split_name}-{split}").is_dir():
                raise MAPSError(
                    f"Split {split} is not available for data group {data_group}."
                )
            else:
                group_path = group_path / f"{self.maps_manager.split_name}-{split}"

        df = pd.read_csv(group_path / "data.tsv", sep="\t")
        json_path = group_path / "maps.json"
        from clinicadl.utils.preprocessing import path_decoder

        with json_path.open(mode="r") as f:
            parameters = json.load(f, object_hook=path_decoder)
        return df, parameters

    def _check_leakage(self, data_group, test_df):
        """
        Checks that no intersection exist between the participants used for training and those used for testing.

        Args:
            data_group (str): name of the data group
            test_df (pd.DataFrame): Table of participant_id / session_id of the data group
        Raises:
            ClinicaDLDataLeakageError: if data_group not in ["train", "validation"] and there is an intersection
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
