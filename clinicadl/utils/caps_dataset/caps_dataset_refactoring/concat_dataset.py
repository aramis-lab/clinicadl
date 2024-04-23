from pathlib import Path
from typing import Iterable, Union

import caps_dataset
import pandas as pd
from torch.utils.data import ConcatDataset, Dataset, StackDataset, dataloader


class CapsConcatDataset(ConcatDataset):
    """Concatenation of CapsDataset"""

    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super().__init__(datasets=datasets)

        mode = [d.mode for d in self.datasets]
        if all(
            i == mode[0] for i in mode
        ):  # check that all the CaspDataset have the same mode
            self.mode = mode[0]
        else:
            raise AttributeError(
                "All the CapsDataset must have the same mode: 'image','patch','roi','slice', etc."
            )


class CapsPairedDataset(StackDataset):
    def __init__(self, datasets: Union[tuple, dict]) -> None:
        self.datasets = list(datasets)  # CapsDatasets list
        self.n_datasets = len(self.datasets)  # number of CapsDatasets
        self.mode = [d.mode for d in self.datasets]  # modes of each CapsDatasets
        super().__init__(*datasets)  # * unpack the tuples


class CapsUnpairedDataset(Dataset):
    r"""Dataset as a stacking of multiple datasets that can have different lengths.

    This class is useful to assemble different parts of complex input data, given as datasets.

    Args:
        datasets (list[Dataset]): Iterable of Datasets.
    """

    def __init__(self, datasets: Union[tuple, list]) -> None:
        self.datasets = list(datasets)  # list of CapsDatasets
        self.n_datasets = len(self.datasets)  # number of CapsDatasets
        self.mode = [d.mode for d in self.datasets]  # modes of each CapsDatasets
        self.len_dataset = [
            d.__len__() for d in self.datasets
        ]  # length of each datasets

    def __getitem__(self, indexes: list):
        # checks that the number of datasets is equal to the number of index
        if len(indexes) == self.n_datasets:
            # checks that the indexes < to the length of each datasets
            if all(indexes[i] < self.len_dataset[i] for i in range(len(indexes))):
                return tuple(
                    [
                        self.datasets[i].__getitem__(indexes[i])
                        for i in range(len(indexes))
                    ]
                )
            else:
                raise ValueError(
                    "The indexes must be inferior than the length of each dataset"
                )
        else:
            raise ValueError(" The number of indexes must match the number of datasets")


def display_controlfile(control_file: pd.DataFrame) -> None:
    import matplotlib.pyplot  # must be imported to display overlay

    def codes_labels(labels_columns: pd.Series):
        categorical_label = pd.CategoricalIndex(labels_columns)
        codes = categorical_label.codes
        return codes

    def make_pretty(styler, gmap=codes_labels(control_file["label_tsv_path"])):
        styler.set_caption("Control File")
        styler.background_gradient(axis=0, cmap="YlGnBu", subset="assembly_id")
        styler.background_gradient(subset="label_tsv_path", gmap=gmap)
        return styler

    return control_file.style.pipe(make_pretty)


def hyper_dataset(control_file_path: Path, paired: bool = False) -> Dataset:
    mandatory_col = [
        "caps_directory",
        "label_tsv_path",
        "preprocessing_json_path",
        "assembly_id",
    ]

    control_file = pd.read_csv(control_file_path, sep="\t")

    # check that the mandatory columns exist
    check_mandatory = True
    for col in mandatory_col:
        check_mandatory = check_mandatory & any(control_file.columns.str.contains(col))
    if not check_mandatory:
        raise ValueError("Mandatory colmuns not recognized")

    # count the number of dataset to stack
    control_file.set_index("assembly_id", inplace=True)
    assembly_id = control_file.index.unique().sort_values()  # sort it before
    number_multicohort = control_file.index.value_counts()
    number_stack = len(assembly_id)

    if not paired:  # unpaired dataset
        DatasetsToStackList = []
        for i in range(number_stack):
            if number_multicohort.loc[assembly_id[i]] > 1:
                DatasetsToConcatList = []

                sub_control_file_i = (
                    control_file.copy().loc[assembly_id[i]].reset_index(drop=True)
                )

                for j in range(len(sub_control_file_i)):
                    caps_arguments_i = sub_control_file_i.loc[j]
                    CapsDataset_i = caps_dataset.CapsDatasetImage(
                        caps_directory=Path(caps_arguments_i["caps_directory"]),
                        tsv_label=Path(caps_arguments_i["label_tsv_path"]),
                        preprocessing_dict=Path(
                            caps_arguments_i["preprocessing_json_path"]
                        ),
                        train_transformations=None,
                        label_presence=False,
                        label=None,
                        label_code=None,
                        all_transformations=None,
                    )

                    DatasetsToConcatList.append(CapsDataset_i)

                CapsConcat_i = CapsConcatDataset(DatasetsToConcatList)
            else:
                sub_control_file_i = control_file.loc[assembly_id[i]]
                CapsConcat_i = caps_dataset.CapsDatasetImage(
                    caps_directory=Path(sub_control_file_i["caps_directory"]),
                    tsv_label=Path(sub_control_file_i["label_tsv_path"]),
                    preprocessing_dict=Path(
                        sub_control_file_i["preprocessing_json_path"]
                    ),
                    train_transformations=None,
                    label_presence=False,
                    label=None,
                    label_code=None,
                    all_transformations=None,
                )

            DatasetsToStackList.append(CapsConcat_i)

        return CapsUnpairedDataset(DatasetsToStackList)
    else:  # paired dataset
        ###########################
        #           check         #
        ###########################

        # check 1, same number of cohort in each columns
        check1 = all(i == number_multicohort[0] for i in number_multicohort)
        if not check1:
            raise ValueError(
                "For the paired dataset, you should have the same number of multicohort number"
            )

        # check 2, each labels_tsv same number than multi modalities
        controlcheck2 = control_file.reset_index().set_index(["label_tsv_path"])
        controlcheck2 = controlcheck2.index.value_counts()
        check2 = all(i == controlcheck2[0] for i in controlcheck2)

        if not check2:
            raise ValueError(
                "Each label file should appears the same number of time for each assembly ID"
            )

        # check 3, equal number of combination assembly_id, labels
        controlcheck3 = control_file.reset_index().set_index(["label_tsv_path"])
        index = controlcheck3.index.unique()
        check3 = all(
            (
                controlcheck3.loc[index[0], "assembly_id"].values
                == controlcheck3.loc[index[i], "assembly_id"]
            ).all()
            for i in range(len(index))
        )

        if not check3:
            raise ValueError(
                "Each combination of assembly ID, label_tsv_path should be equal"
            )

        ###########################
        #      compute paired     #
        ###########################

        DatasetsToStackList = []
        for i in range(number_stack):
            if number_multicohort.loc[assembly_id[i]] > 1:
                DatasetsToConcatList = []

                sub_control_file_i = (
                    control_file.copy()
                    .loc[assembly_id[i]]
                    .reset_index(drop=True)
                    .set_index("label_tsv_path")
                )

                for j in range(len(index)):
                    # we use the index by unicity, so they will be called in the same order at for each assembly ID which means
                    # that we ensure the same construction for the concatenation
                    caps_arguments_i = sub_control_file_i.loc[index[j]]
                    CapsDataset_i = caps_dataset.CapsDatasetImage(
                        caps_directory=Path(caps_arguments_i["caps_directory"]),
                        tsv_label=Path(index[j]),
                        preprocessing_dict=Path(
                            caps_arguments_i["preprocessing_json_path"]
                        ),
                        train_transformations=None,
                        label_presence=False,
                        label=None,
                        label_code=None,
                        all_transformations=None,
                    )

                    DatasetsToConcatList.append(CapsDataset_i)

                CapsConcat_i = CapsConcatDataset(DatasetsToConcatList)
            else:
                sub_control_file_i = control_file.loc[assembly_id[i]]
                CapsConcat_i = caps_dataset.CapsDatasetImage(
                    caps_directory=Path(sub_control_file_i["caps_directory"]),
                    tsv_label=Path(sub_control_file_i["label_tsv_path"]),
                    preprocessing_dict=Path(
                        sub_control_file_i["preprocessing_json_path"]
                    ),
                    train_transformations=None,
                    label_presence=False,
                    label=None,
                    label_code=None,
                    all_transformations=None,
                )

            DatasetsToStackList.append(CapsConcat_i)
        return CapsPairedDataset(DatasetsToStackList)


# from typing import List, Optional
# class DataloaderCaps(dataloader):

#     def __init__(self,
#                  dataset: Dataset, batch_size: List[int] = 1,
#                  shuffle: Optional[bool] = None, sampler: Union[Sampler, Iterable, None] = None,
#                  batch_sampler: Union[Sampler[List], Iterable[List], None] = None,
#                  num_workers: int = 0, collate_fn: Optional[_collate_fn_t] = None,
#                  pin_memory: bool = False, drop_last: bool = False,
#                  timeout: float = 0, worker_init_fn: Optional[_worker_init_fn_t] = None,
#                  multiprocessing_context=None, generator=None,
#                  *, prefetch_factor: Optional[int] = None,
#                  persistent_workers: bool = False,
#                  pin_memory_device: str = ""):

#         self.dataset = dataset
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#         self.sampler = sampler


#         for i in range(dataset.n_datasets):

#             dataloader_i = Dataloader
