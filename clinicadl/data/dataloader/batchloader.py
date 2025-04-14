from typing import Any, List, Union

import torch
import torchio as tio

from clinicadl.transforms.extraction import Sample


class BatchLoader(list):
    """
    A class to load and manage batches of samples.

    Attributes:
        samples (list[Sample]): A list of Sample objects that make up the batch.
    """

    def __init__(self, samples: list[Sample]):
        """
        Initialize the BatchLoader with a list of samples.

        Parameters
        ----------
            samples: list[Sample]
                A list of Sample objects to be loaded into the batch.

        Raises
        ------
            ValueError: If the provided list of samples is empty.
        """
        # Store the samples of the batch
        super().__init__(samples)

        if len(self) == 0:
            raise ValueError("No samples to load.")

    def get_images(self) -> torch.Tensor:
        """
        Get the images from the samples in the batch.

        Returns
        -------
            torch.Tensor: A tensor containing all the images from the batch.
        """
        # Return the images of the batch
        return torch.stack([sample.image.tensor for sample in self], dim=0)

    def get_labels(self) -> Union[torch.Tensor, List[Any]]:
        """
        Gets the labels from the samples in the batch.

        Returns
        -------
        Union[torch.Tensor, List[Any]]
            A tensor or a list containing all the labels from the batch.
            It will be a list if the labels are heterogeneous (e.g. a mask and a scalar) or if any
            of the label is ``None``. Otherwise, it will be a tensor.
        """
        labels = [
            sample.label.tensor
            if isinstance(sample.label, tio.LabelMap)
            else sample.label
            for sample in self
        ]
        if all(isinstance(label, torch.Tensor) for label in labels):
            return torch.stack(labels, dim=0)
        elif all(isinstance(label, (int, float)) for label in labels):
            return torch.tensor(
                labels,
                dtype=torch.float32,
            )
        else:
            return labels
