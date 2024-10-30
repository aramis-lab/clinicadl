from clinicadl.dataset.caps_dataset import CapsDataset


class ConcatDataset(CapsDataset):
    def __init__(self, list_: list[CapsDataset]):
        """TO COMPLETE"""
