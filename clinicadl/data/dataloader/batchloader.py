import torch

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
        return torch.cat([sample.sample for sample in self], dim=0).unsqueeze(1)

    def get_labels(self) -> torch.Tensor:
        """
        Get the labels from the samples in the batch.

        Returns
        -------
            torch.Tensor: A tensor containing all the labels from the batch.
        """
        # Return the labels of the batch
        if all(isinstance(sample.label, torch.Tensor) for sample in self):
            list_ = []
            for sample in self:
                list_.append(sample.label)
            return torch.cat(list_, dim=0).unsqueeze(1)
        else:
            return torch.tensor(
                [sample.label if sample.label else sample.sample for sample in self],
                dtype=torch.float32,
            ).unsqueeze(1)  # TODO: check torch.long
