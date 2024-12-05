from copy import deepcopy
from typing import Sequence, Tuple

from clinicadl.dataset.datasets.caps_dataset import CapsDataset


class Subset(CapsDataset):
    """
    A subset of a caps dataset, build from a list of (subject, session).
    Won't work like this because CapsDataset is an abstract class.
    Maybe a function is enough?
    """

    def __init__(
        self, dataset: CapsDataset, subjects_sessions: Sequence[Tuple[str, str]]
    ) -> None:
        subset_indices = []
        for i, row in enumerate(dataset.df.iterrows()):
            if (row["participant_id"], row["session_id"]) in subjects_sessions:
                subset_indices.append(i)

        config = deepcopy(dataset.config)
        config.data.data_df = dataset.df.iloc[subset_indices]
        super().__init__(
            config=config,
            label_presence=dataset.label_presence,
            preprocessing_dict=dataset.preprocessing_dict,
        )

        self.base_dataset = dataset
        self.subjects_sessions = subjects_sessions
