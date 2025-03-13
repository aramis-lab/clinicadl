import torch

from clinicadl.transforms.extraction import Sample


class BatchLoader:
    def __init__(self, samples: list[Sample]):
        # Stocker les échantillons du batch
        self.samples = samples

        if len(self) == 0:
            raise ValueError("No samples to load.")

    def __len__(self):
        # Retourner la taille du batch
        return len(self.samples)

    def get_images(self):
        # Retourner les images du batch
        return torch.cat([sample.sample for sample in self.samples], dim=0).unsqueeze(1)

    def get_labels(self):
        # Retourner les labels du batch
        if all(isinstance(sample.label, torch.Tensor) for sample in self.samples):
            list_ = []
            for sample in self.samples:
                list_.append(sample.label)
            return torch.cat(list_, dim=0).unsqueeze(1)
        else:
            return torch.tensor(
                [sample.label for sample in self.samples], dtype=torch.float32
            ).unsqueeze(1)  # TODO: check torch.long

    def __getitem__(self, key: int) -> Sample:
        # Retourner un élément du batch à l'indice 'key'
        return self.samples[key]
