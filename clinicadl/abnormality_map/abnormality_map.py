import json
import numpy as np
import nibabel as nib
from pathlib import Path
from copy import deepcopy
from typing import Tuple, Callable, Union

import torch
from torch.utils.data import DataLoader

from clinicadl.utils.caps_dataset.data import load_data_test, return_dataset, get_transforms
from clinicadl.utils.maps_manager import MapsManager
from clinicadl.utils.maps_manager.maps_manager_utils import read_json


def residual_fn(
    x: torch.Tensor,
    x_hat: torch.Tensor,
) -> Tuple[torch.Tensor, str]:
    return x - x_hat, "residual"

# def residual_fn(
#     x: nib.Nifti1Image,
#     x_hat: nib.Nifti1Image,
# ) -> Tuple[nib.Nifti1Image, str]:
#     affine = x.affine
#     x_data = x.get_fdata().numpy()
#     x_hat_data = x_hat.get_fdata().numpy()
#     return nib.Nifti1Image(x_data - x_hat_data, eye(4)), "residual"

abn_map_dict = {
    "residual": residual_fn,
}

def compute_abnormality_map(
    maps_dir: Path,
    preprocessing_json: Path,
    data_group: str,
    gpu: bool,
    n_proc: int,
    batch_size: int,
    abn_map_fn: Union[str, Callable[[nib.Nifti1Image, nib.Nifti1Image], Tuple[nib.Nifti1Image, str]]] = residual_fn,
    split_list: list[int] = None,
    tsv_path: Path = None,
    selection_metrics: str = None,
    multi_cohort: bool = False,
):
    """
    Computes the abnormality map for a given split and data group using the model selected with the selection metrics.
    
    Args:
        maps_manager (MapsManager): MAPS manager object
        split_list (list[int]): list of splits to compute the abnormality map on
        data_group (str): data group to compute the abnormality map on
        selection_metrics (str): selection metrics to use to select the best model
        abn_map_fn (Callable[[nib.Nifti1Image, nib.Nifti1Image], Tuple[nib.Nifti1Image, str]], optional): 
            method to compute the abnormality map. Defaults to residual_fn.
    """
        
    maps_manager = MapsManager(maps_dir)
    
    if not split_list:
        split_list = maps_manager._find_splits()
        
    for split in split_list:
        
        if tsv_path is not None:
            group_df = load_data_test(
                tsv_path,
                "CN",
                multi_cohort=multi_cohort,
            )
        
        group_df, group_parameters = maps_manager.get_group_info(data_group, split)

        if not selection_metrics:
            split_selection_metrics = maps_manager._find_selection_metrics(split)[0]
        else:
            split_selection_metrics = selection_metrics[0]
            
        preprocessing_dict = read_json(preprocessing_json)

        input_preprocessing_dict = deepcopy(preprocessing_dict)
        input_pattern = preprocessing_dict["file_type"]["pattern"].replace("_mood.pt", "_input.pt")
        input_preprocessing_dict["file_type"]["pattern"] = input_pattern
        input_preprocessing_dict["preprocessing"] = "custom"
        
        output_preprocessing_dict = deepcopy(preprocessing_dict)
        output_pattern = preprocessing_dict["file_type"]["pattern"].replace("_mood.pt", "_output.pt")
        output_preprocessing_dict["file_type"]["pattern"] = output_pattern
        output_preprocessing_dict["preprocessing"] = "custom"
        
        # TO DO: code to retrieve transforms
        # _, all_transforms = get_transforms(
        #     normalize=maps_manager.normalize,
        #     data_augmentation=maps_manager.data_augmentation,
        #     size_reduction=maps_manager.size_reduction,
        #     size_reduction_factor=maps_manager.size_reduction_factor,
        # )

        caps_output_dir = Path(maps_dir, f"split-{split}", f"best-{split_selection_metrics}", "CapsOutput")
        
        input_data = return_dataset(
            caps_output_dir,
            group_df,
            input_preprocessing_dict,
            all_transformations=None, # TO DO: change this
            multi_cohort=group_parameters["multi_cohort"],
        )

        output_data = return_dataset(
            caps_output_dir,
            group_df,
            output_preprocessing_dict,
            all_transformations=None, # TO DO: change this
            multi_cohort=group_parameters["multi_cohort"],
        )

        input_loader = DataLoader(
            input_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=n_proc,
        )
        
        output_loader = DataLoader(
            output_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=n_proc,
        )
        
        for i, (input_batch, output_batch) in enumerate(zip(input_loader, output_loader)):
            
            for idx in range(len(input_batch['participant_id'])):
                
                assert input_batch['participant_id'][idx] == output_batch['participant_id'][idx]

                participant_id = input_batch['participant_id'][idx]
                session_id = input_batch['session_id'][idx]
                
                if isinstance(abn_map_fn, str):
                    abn_map_fn = abn_map_dict[abn_map_fn]

                abn_map_pt, abn_map_pattern = abn_map_fn(output_batch['data'][idx], input_batch['data'][idx])
                abn_map_nii = nib.Nifti1Image(abn_map_pt.numpy(), np.eye(4))

                abn_map_pt_path = input_batch['image_path'][idx].replace("_input.pt", f"_{abn_map_pattern}.pt")
                abn_map_nii_path = Path(caps_output_dir, "subjects", participant_id, session_id, "custom", f"{participant_id}_{session_id}_{abn_map_pattern}.nii.gz")

                torch.save(abn_map_pt, abn_map_pt_path)
                nib.save(abn_map_nii, abn_map_nii_path)

                print("Saved abnormality map at", abn_map_pt_path)
                print("Saved abnormality map at", abn_map_nii_path)
                    
        abn_map_preprocessing_dict = deepcopy(preprocessing_dict)
        abn_map_preprocessing_dict["file_type"]["pattern"] = abn_map_pattern
            
        abn_map_preprocessing_json = json.dumps(abn_map_preprocessing_dict, skipkeys=True, indent=4)
        f = open(preprocessing_json.replace(".json", f"_{abn_map_pattern}.json"), "w")
        f.write(abn_map_preprocessing_json)
        f.close()