import json
import shutil
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from clinicadl.config.config.pipelines.interpret import InterpretConfig
from clinicadl.config.config.pipelines.predict import PredictConfig
from clinicadl.transforms.transforms import get_transforms
from clinicadl.utils.caps_dataset.data import (
    return_dataset,
)
from clinicadl.utils.exceptions import (
    ClinicaDLArgumentError,
    ClinicaDLDataLeakageError,
    MAPSError,
)
from clinicadl.utils.maps_manager.ddp import DDP, cluster
from clinicadl.utils.maps_manager.maps_manager import MapsManager

logger = getLogger("clinicadl.predict_manager")
level_list: List[str] = ["warning", "info", "debug"]


class PredictManager:
    def __init__(self, _config: Union[PredictConfig, InterpretConfig]) -> None:
        self.maps_manager = MapsManager(_config.maps_dir)
        self._config = _config

    def predict(
        self,
        label_code: Union[str, dict[str, int]] = "default",
    ):
        """Performs the prediction task on a subset of caps_directory defined in a TSV file.
        Parameters
        ----------
        data_group : str
            name of the data group tested.
        caps_directory : Path (optional, default=None)
            path to the CAPS folder. For more information please refer to
            [clinica documentation](https://aramislab.paris.inria.fr/clinica/docs/public/latest/CAPS/Introduction/).
            Default will load the value of an existing data group
        tsv_path : Path (optional, default=None)
            path to a TSV file containing the list of participants and sessions to test.
            Default will load the DataFrame of an existing data group
        split_list : List[int] (optional, default=None)
            list of splits to test. Default perform prediction on all splits available.
        selection_metrics : List[str] (optional, default=None)
            list of selection metrics to test.
                Default performs the prediction on all selection metrics available.
        multi_cohort : bool (optional, default=False)
            If True considers that tsv_path is the path to a multi-cohort TSV.
        diagnoses : List[str] (optional, default=())
            List of diagnoses to load if tsv_path is a split_directory.
            Default uses the same as in training step.
        use_labels : bool (optional, default=True)
            If True, the labels must exist in test meta-data and metrics are computed.
        batch_size : int (optional, default=None)
            If given, sets the value of batch_size, else use the same as in training step.
        n_proc : int (optional, default=None)
            If given, sets the value of num_workers, else use the same as in training step.
        gpu : bool (optional, default=None)
            If given, a new value for the device of the model will be computed.
        amp : bool (optional, default=False)
            If enabled, uses Automatic Mixed Precision (requires GPU usage).
        overwrite : bool (optional, default=False)
            If True erase the occurrences of data_group.
        label : str (optional, default=None)
            Target label used for training (if network_task in [`regression`, `classification`]).
        label_code : Optional[Dict[str, int]] (optional, default="default")
            dictionary linking the target values to a node number.
        save_tensor : bool (optional, default=False)
            If true, save the tensor predicted for reconstruction task
        save_nifti : bool (optional, default=False)
            If true, save the nifti associated to the prediction for reconstruction task.
        save_latent_tensor : bool (optional, default=False)
            If true, save the tensor from the latent space for reconstruction task.
        skip_leak_check : bool (optional, default=False)
            If true, skip the leak check (not recommended).
        Examples
        --------
        >>> _input_
        _output_
        """

        assert isinstance(self._config, PredictConfig)

        self._config.check_output_saving_nifti(self.maps_manager.network_task)
        self._config.adapt_data_with_maps_manager_info(self.maps_manager)
        self._config.adapt_dataloader_with_maps_manager_info(self.maps_manager)
        self._config.adapt_cross_val_with_maps_manager_info(self.maps_manager)
        self._config.check_output_saving_tensor(self.maps_manager.network_task)

        _, all_transforms = get_transforms(
            normalize=self.maps_manager.normalize,
            data_augmentation=self.maps_manager.data_augmentation,
            size_reduction=self.maps_manager.size_reduction,
            size_reduction_factor=self.maps_manager.size_reduction_factor,
        )
        group_df = self._config.create_groupe_df()
        self._check_data_group(group_df)
        criterion = self.maps_manager.task_manager.get_criterion(self.maps_manager.loss)
        self._check_data_group(df=group_df)

        assert self._config.split  # don't know if needed ? try to raise an exception ?
        # assert self._config.label

        for split in self._config.split:
            logger.info(f"Prediction of split {split}")
            group_df, group_parameters = self.get_group_info(
                self._config.data_group, split
            )
            # Find label code if not given
            if self._config.is_given_label_code(self.maps_manager.label, label_code):
                self.maps_manager.task_manager.generate_label_code(
                    group_df, self._config.label
                )
            # Erase previous TSV files on master process
            if not self._config.selection_metrics:
                split_selection_metrics = self.maps_manager._find_selection_metrics(
                    split
                )
            else:
                split_selection_metrics = self._config.selection_metrics
            for selection in split_selection_metrics:
                tsv_dir = (
                    self.maps_manager.maps_path
                    / f"{self.maps_manager.split_name}-{split}"
                    / f"best-{selection}"
                    / self._config.data_group
                )
                tsv_pattern = f"{self._config.data_group}*.tsv"
                for tsv_file in tsv_dir.glob(tsv_pattern):
                    tsv_file.unlink()
            self._config.check_label(self.maps_manager.label)
            if self.maps_manager.multi_network:
                self._predict_multi(
                    group_parameters,
                    group_df,
                    all_transforms,
                    label_code,
                    criterion,
                    split,
                    split_selection_metrics,
                )
            else:
                self._predict_single(
                    group_parameters,
                    group_df,
                    all_transforms,
                    label_code,
                    criterion,
                    split,
                    split_selection_metrics,
                )
            if cluster.master:
                self.maps_manager._ensemble_prediction(
                    self._config.data_group,
                    split,
                    self._config.selection_metrics,
                    self._config.use_labels,
                    self._config.skip_leak_check,
                )

    def _predict_multi(
        self,
        group_parameters,
        group_df,
        all_transforms,
        label_code,
        criterion,
        split,
        split_selection_metrics,
    ):
        """_summary_
        Parameters
        ----------
        group_parameters : _type_
            _description_
        group_df : _type_
            _description_
        all_transforms : _type_
            _description_
        use_labels : _type_
            _description_
        label : _type_
            _description_
        label_code : _type_
            _description_
        batch_size : _type_
            _description_
        n_proc : _type_
            _description_
        criterion : _type_
            _description_
        data_group : _type_
            _description_
        split : _type_
            _description_
        split_selection_metrics : _type_
            _description_
        gpu : _type_
            _description_
        amp : _type_
            _description_
        save_tensor : _type_
            _description_
        save_latent_tensor : _type_
            _description_
        save_nifti : _type_
            _description_
        selection_metrics : _type_
            _description_
        Examples
        --------
        >>> _input_
        _output_
        Notes
        -----
        _notes_
        See Also
        --------
        - _related_
        """
        assert isinstance(self._config, PredictConfig)
        # assert self._config.label

        for network in range(self.maps_manager.num_networks):
            data_test = return_dataset(
                group_parameters["caps_directory"],
                group_df,
                self.maps_manager.preprocessing_dict,
                all_transformations=all_transforms,
                multi_cohort=group_parameters["multi_cohort"],
                label_presence=self._config.use_labels,
                label=self._config.label,
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
                    self._config.batch_size
                    if self._config.batch_size is not None
                    else self.maps_manager.batch_size
                ),
                shuffle=False,
                sampler=DistributedSampler(
                    data_test,
                    num_replicas=cluster.world_size,
                    rank=cluster.rank,
                    shuffle=False,
                ),
                num_workers=self._config.n_proc
                if self._config.n_proc is not None
                else self.maps_manager.n_proc,
            )
            self.maps_manager._test_loader(
                test_loader,
                criterion,
                self._config.data_group,
                split,
                split_selection_metrics,
                use_labels=self._config.use_labels,
                gpu=self._config.gpu,
                amp=self._config.amp,
                network=network,
            )
            if self._config.save_tensor:
                logger.debug("Saving tensors")
                self.maps_manager._compute_output_tensors(
                    data_test,
                    self._config.data_group,
                    split,
                    self._config.selection_metrics,
                    gpu=self._config.gpu,
                    network=network,
                )
            if self._config.save_nifti:
                self._compute_output_nifti(
                    data_test,
                    split,
                    network=network,
                )
            if self._config.save_latent_tensor:
                self._compute_latent_tensors(
                    dataset=data_test,
                    split=split,
                    network=network,
                )

    def _predict_single(
        self,
        group_parameters,
        group_df,
        all_transforms,
        label_code,
        criterion,
        split,
        split_selection_metrics,
    ):
        """_summary_
        Parameters
        ----------
        group_parameters : _type_
            _description_
        group_df : _type_
            _description_
        all_transforms : _type_
            _description_
        use_labels : _type_
            _description_
        label : _type_
            _description_
        label_code : _type_
            _description_
        batch_size : _type_
            _description_
        n_proc : _type_
            _description_
        criterion : _type_
            _description_
        data_group : _type_
            _description_
        split : _type_
            _description_
        split_selection_metrics : _type_
            _description_
        gpu : _type_
            _description_
        amp : _type_
            _description_
        save_tensor : _type_
            _description_
        save_latent_tensor : _type_
            _description_
        save_nifti : _type_
            _description_
        selection_metrics : _type_
            _description_
        Examples
        --------
        >>> _input_
        _output_
        Notes
        -----
        _notes_
        See Also
        --------
        - _related_
        """

        assert isinstance(self._config, PredictConfig)
        # assert self._config.label

        data_test = return_dataset(
            group_parameters["caps_directory"],
            group_df,
            self.maps_manager.preprocessing_dict,
            all_transformations=all_transforms,
            multi_cohort=group_parameters["multi_cohort"],
            label_presence=self._config.use_labels,
            label=self._config.label,
            label_code=(
                self.maps_manager.label_code if label_code == "default" else label_code
            ),
        )
        test_loader = DataLoader(
            data_test,
            batch_size=(
                self._config.batch_size
                if self._config.batch_size is not None
                else self.maps_manager.batch_size
            ),
            shuffle=False,
            sampler=DistributedSampler(
                data_test,
                num_replicas=cluster.world_size,
                rank=cluster.rank,
                shuffle=False,
            ),
            num_workers=self._config.n_proc
            if self._config.n_proc is not None
            else self.maps_manager.n_proc,
        )
        self.maps_manager._test_loader(
            test_loader,
            criterion,
            self._config.data_group,
            split,
            split_selection_metrics,
            use_labels=self._config.use_labels,
            gpu=self._config.gpu,
            amp=self._config.amp,
        )
        if self._config.save_tensor:
            logger.debug("Saving tensors")
            self.maps_manager._compute_output_tensors(
                data_test,
                self._config.data_group,
                split,
                self._config.selection_metrics,
                gpu=self._config.gpu,
            )
        if self._config.save_nifti:
            self._compute_output_nifti(
                data_test,
                split,
            )
        if self._config.save_latent_tensor:
            self._compute_latent_tensors(
                dataset=data_test,
                split=split,
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
        for selection_metric in self._config.selection_metrics:
            # load the best trained model during the training
            model, _ = self.maps_manager._init_model(
                transfer_path=self.maps_manager.maps_path,
                split=split,
                transfer_selection=selection_metric,
                gpu=self._config.gpu,
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
                / self._config.data_group
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

        for selection_metric in self._config.selection_metrics:
            # load the best trained model during the training
            model, _ = self.maps_manager._init_model(
                transfer_path=self.maps_manager.maps_path,
                split=split,
                transfer_selection=selection_metric,
                gpu=self._config.gpu,
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
                / self._config.data_group
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

    def interpret(self):
        """Performs the interpretation task on a subset of caps_directory defined in a TSV file.
        The mean interpretation is always saved, to save the individual interpretations set save_individual to True.
        Parameters
        ----------
        data_group : str
            Name of the data group interpreted.
        name : str
            Name of the interpretation procedure.
        method : str
            Method used for extraction (ex: gradients, grad-cam...).
        caps_directory : Path (optional, default=None)
            Path to the CAPS folder. For more information please refer to
            [clinica documentation](https://aramislab.paris.inria.fr/clinica/docs/public/latest/CAPS/Introduction/).
            Default will load the value of an existing data group.
        tsv_path : Path (optional, default=None)
            Path to a TSV file containing the list of participants and sessions to test.
            Default will load the DataFrame of an existing data group.
        split_list : list[int] (optional, default=None)
            List of splits to interpret. Default perform interpretation on all splits available.
        selection_metrics : list[str] (optional, default=None)
            List of selection metrics to interpret.
            Default performs the interpretation on all selection metrics available.
        multi_cohort : bool (optional, default=False)
            If True considers that tsv_path is the path to a multi-cohort TSV.
        diagnoses : list[str] (optional, default=())
            List of diagnoses to load if tsv_path is a split_directory.
            Default uses the same as in training step.
        target_node : int (optional, default=0)
            Node from which the interpretation is computed.
        save_individual : bool (optional, default=False)
            If True saves the individual map of each participant / session couple.
        batch_size : int (optional, default=None)
            If given, sets the value of batch_size, else use the same as in training step.
        n_proc : int (optional, default=None)
            If given, sets the value of num_workers, else use the same as in training step.
        gpu : bool (optional, default=None)
            If given, a new value for the device of the model will be computed.
        amp : bool (optional, default=False)
            If enabled, uses Automatic Mixed Precision (requires GPU usage).
        overwrite : bool (optional, default=False)
            If True erase the occurrences of data_group.
        overwrite_name : bool (optional, default=False)
            If True erase the occurrences of name.
        level : int (optional, default=None)
            Layer number in the convolutional part after which the feature map is chosen.
        save_nifti : bool (optional, default=False)
            If True, save the interpretation map in nifti format.
        Raises
        ------
        NotImplementedError
            If the method is not implemented
        NotImplementedError
            If the interpretaion of multi network is asked
        MAPSError
            If the interpretation has already been determined.
        """
        assert isinstance(self._config, InterpretConfig)

        self._config.adapt_data_with_maps_manager_info(self.maps_manager)
        self._config.adapt_dataloader_with_maps_manager_info(self.maps_manager)
        self._config.adapt_cross_val_with_maps_manager_info(self.maps_manager)

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
        group_df = self._config.create_groupe_df()
        self._check_data_group(group_df)

        assert self._config.split
        for split in self._config.split:
            logger.info(f"Interpretation of split {split}")
            df_group, parameters_group = self.get_group_info(
                self._config.data_group, split
            )
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
                batch_size=self._config.batch_size,
                shuffle=False,
                num_workers=self._config.n_proc,
            )
            if not self._config.selection_metrics:
                self._config.selection_metrics = (
                    self.maps_manager._find_selection_metrics(split)
                )
            for selection_metric in self._config.selection_metrics:
                logger.info(f"Interpretation of metric {selection_metric}")
                results_path = (
                    self.maps_manager.maps_path
                    / f"{self.maps_manager.split_name}-{split}"
                    / f"best-{selection_metric}"
                    / self._config.data_group
                    / f"interpret-{self._config.name}"
                )
                if (results_path).is_dir():
                    if self._config.overwrite_name:
                        shutil.rmtree(results_path)
                    else:
                        raise MAPSError(
                            f"Interpretation name {self._config.name} is already written. "
                            f"Please choose another name or set overwrite_name to True."
                        )
                results_path.mkdir(parents=True)
                model, _ = self.maps_manager._init_model(
                    transfer_path=self.maps_manager.maps_path,
                    split=split,
                    transfer_selection=selection_metric,
                    gpu=self._config.gpu,
                )
                interpreter = self._config.get_method()(model)
                cum_maps = [0] * data_test.elem_per_image
                for data in test_loader:
                    images = data["image"].to(model.device)
                    map_pt = interpreter.generate_gradients(
                        images,
                        self._config.target_node,
                        level=self._config.level,
                        amp=self._config.amp,
                    )
                    for i in range(len(data["participant_id"])):
                        mode_id = data[f"{self.maps_manager.mode}_id"][i]
                        cum_maps[mode_id] += map_pt[i]
                        if self._config.save_individual:
                            single_path = (
                                results_path
                                / f"{data['participant_id'][i]}_{data['session_id'][i]}_{self.maps_manager.mode}-{data[f'{self.maps_manager.mode}_id'][i]}_map.pt"
                            )
                            torch.save(map_pt[i], single_path)
                            if self._config.save_nifti:
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
                    if self._config.save_nifti:
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
        df: Optional[pd.DataFrame] = None,
    ):
        """Check if a data group is already available if other arguments are None.
        Else creates a new data_group.

        Parameters
        ----------
        data_group : str
            name of the data group
        caps_directory : str (optional, default=None)
            input CAPS directory
        df : pd.DataFrame (optional, default=None)
            Table of participant_id / session_id of the data group
        multi_cohort : bool (optional, default=False)
            indicates if the input data comes from several CAPS
        overwrite : bool (optional, default=False)
            If True former definition of data group is erased
        label : str (optional, default=None)
            label name if applicable
        split_list : list[int] (optional, default=None)
            _description_
        skip_leak_check : bool (optional, default=False)
            _description_

        Raises
        ------
        MAPSError
            when trying to overwrite train or validation data groups
        ClinicaDLArgumentError
            when caps_directory or df are given but data group already exists
        ClinicaDLArgumentError
            when caps_directory or df are not given and data group does not exist

        """
        group_dir = self.maps_manager.maps_path / "groups" / self._config.data_group
        logger.debug(f"Group path {group_dir}")
        if group_dir.is_dir():  # Data group already exists
            if self._config.overwrite:
                if self._config.data_group in ["train", "validation"]:
                    raise MAPSError("Cannot overwrite train or validation data group.")
                else:
                    # if not split_list:
                    #     split_list = self.maps_manager._find_splits()
                    assert self._config.split
                    for split in self._config.split:
                        selection_metrics = self.maps_manager._find_selection_metrics(
                            split
                        )
                        for selection in selection_metrics:
                            results_path = (
                                self.maps_manager.maps_path
                                / f"{self.maps_manager.split_name}-{split}"
                                / f"best-{selection}"
                                / self._config.data_group
                            )
                            if results_path.is_dir():
                                shutil.rmtree(results_path)
            elif df is not None or (
                self._config.caps_directory is not None
                and self._config.caps_directory != Path("")
            ):
                raise ClinicaDLArgumentError(
                    f"Data group {self._config.data_group} is already defined. "
                    f"Please do not give any caps_directory, tsv_path or multi_cohort to use it. "
                    f"To erase {self._config.data_group} please set overwrite to True."
                )

        elif not group_dir.is_dir() and (
            self._config.caps_directory is None or df is None
        ):  # Data group does not exist yet / was overwritten + missing data
            raise ClinicaDLArgumentError(
                f"The data group {self._config.data_group} does not already exist. "
                f"Please specify a caps_directory and a tsv_path to create this data group."
            )
        elif (
            not group_dir.is_dir()
        ):  # Data group does not exist yet / was overwritten + all data is provided
            if self._config.skip_leak_check:
                logger.info("Skipping data leakage check")
            else:
                self._check_leakage(self._config.data_group, df)
            self._write_data_group(
                self._config.data_group,
                df,
                self._config.caps_directory,
                self._config.multi_cohort,
                label=self._config.label,
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
            elif not (group_path / f"{self.maps_manager.split_name}-{split}").is_dir():
                raise MAPSError(
                    f"Split {split} is not available for data group {data_group}."
                )
            else:
                group_path = group_path / f"{self.maps_manager.split_name}-{split}"

        df = pd.read_csv(group_path / "data.tsv", sep="\t")
        json_path = group_path / "maps.json"
        from clinicadl.preprocessing.preprocessing import path_decoder

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
        if self._config.label in df.columns.values:
            columns += [self._config.label]
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

        selection_metric = self.maps_manager._check_selection_metric(
            split, selection_metric
        )
        if verbose:
            self.maps_manager._print_description_log(
                data_group, split, selection_metric
            )
        map_dir = (
            self.maps_manager.maps_path
            / f"{self.maps_manager.split_name}-{split}"
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
                map_dir / f"mean_{self.maps_manager.mode}-{mode_id}_map.pt"
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
                / f"{participant_id}_{session_id}_{self.maps_manager.mode}-{mode_id}_map.pt"
            )
        return map_pt
