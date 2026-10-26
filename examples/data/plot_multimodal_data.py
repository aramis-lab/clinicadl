"""
Multimodal data example
=======================
"""

# %%
# In this example, we show how to create a dataset and a dataloader for multimodal data. We will use the BidsDLBS dataset, which contains T1w and PET images.

# %%
# T1w dataset
# -----------
# Let us first create a dataset for the T1w images.

from clinicadl.data.datasets.examples import BidsDLBSSmall

t1_dataset = BidsDLBSSmall(pet=False, columns=["Sex", "AgeMRI", "HandednessScore"])


# %%
# Get the first sample of the T1 dataset:
t1_dataset[0]

# %%
# Get information about the image of the first sample of the T1 dataset:
t1_dataset[0].image

# %%
# Notice the shape of the image is (1, 127, 127, 80), its spacing (2.00, 2.00, 2.00), and its orientation PSR+.

# %%
# We can plot the image of the first sample of the T1 dataset:

t1_dataset[0].image.plot()


# %%
# PET dataset
# -----------
# Let us now create a dataset for the PET images.

pet_dataset = BidsDLBSSmall(pet=True, columns=["Sex", "AgePETAmy", "HandednessScore"])


# %%
# Similarly, we can obtain get the first sample of the PET dataset:

pet_dataset[0]

# %%
# We can also retrieve information about the image of the first sample of this PET dataset:

pet_dataset[0].image

# %%
# Notice the shape of the image is now (1, 128, 128, 63), its spacing (2.57, 2.57, 2.43), and its orientation LAS+.

# %%
# We can plot the image of the first sample of the PET dataset:

pet_dataset[0].image.plot()

# %%
# Paired T1w and PET dataset
# --------------------------

# %%
# Uniformize both datasets
# ~~~~~~~~~~~~~~~~~~~~~~~~
# Since the T1w and PET images have different shapes and orientations, we will need to resample them to the same shape, and change their orientations before feeding them to a model. For this, we would need to co-register the images. Thankfully, this was already done in the CapsDLBS dataset, using Clinica, specifically the t1-linear pipeline and the pet-linear pipeline.

from clinicadl.data.datasets.examples import CapsDLBS

t1_caps = CapsDLBS(pet=False, columns=["Sex", "AgeMRI", "HandednessScore"])

pet_caps = CapsDLBS(pet=True, columns=["Sex", "AgePETAmy", "HandednessScore"])

# %%

t1_caps[0].image, pet_caps[0].image

# %%
# Notice the shape of the images is now (1, 169, 208, 179), their spacing (1.00, 1.00, 1.00) and their orientation: RAS+

# %%
t1_caps[0].image.plot()

# %%
pet_caps[0].image.plot()

# %%
# PairedDataset
# ~~~~~~~~~~~~~
# Let us now create the paired T1w and PET dataset.

from clinicadl.data.datasets import PairedDataset

paired_t1_pet_dataset = PairedDataset(
    datasets=[t1_caps, pet_caps],
)

# %%
# Create dataloader
# ~~~~~~~~~~~~~~~~~
# We can now create a dataloader for the paired T1w and PET dataset.

from clinicadl.data.dataloader import DataLoader

# %%
# There are multiple ways to collate the data in a batch.
# %%
# ToBatchCollate
# ^^^^^^^^^^^^^^
from clinicadl.data.dataloader.collate import ToBatchCollate

dataloader = DataLoader(
    paired_t1_pet_dataset, batch_size=2, shuffle=False, collate_fn=ToBatchCollate()
)

batch = next(iter(dataloader))

batch

# %%
# We now have the following:

sub1_t1, sub1_pet = batch[0]
sub2_t1, sub2_pet = batch[1]

(
    sub1_t1.participant_id,
    sub1_t1,
    sub1_pet.participant_id,
    sub1_pet,
    sub2_t1.participant_id,
    sub2_t1,
    sub2_pet.participant_id,
    sub2_pet,
)

# %%
# ToBatchesCollate
# ^^^^^^^^^^^^^^^^

from clinicadl.data.dataloader.collate import ToBatchesCollate

dataloader = DataLoader(
    paired_t1_pet_dataset, batch_size=2, shuffle=False, collate_fn=ToBatchesCollate()
)

batch = next(iter(dataloader))

batch

# %%
# We now have the following:

sub1_t1, sub2_t1 = batch[0]
sub1_pet, sub2_pet = batch[1]

(
    sub1_t1.participant_id,
    sub1_t1,
    sub1_pet.participant_id,
    sub1_pet,
    sub2_t1.participant_id,
    sub2_t1,
    sub2_pet.participant_id,
    sub2_pet,
)

# %%
# MergeBatchesCollate
# ^^^^^^^^^^^^^^^^^^^

from clinicadl.data.dataloader.collate import MergeBatchesCollate

dataloader = DataLoader(
    paired_t1_pet_dataset, batch_size=2, shuffle=False, collate_fn=MergeBatchesCollate()
)

batch = next(iter(dataloader))

batch

# %%
# We now have the following:

sub1_t1_pet = batch[0]
sub2_t1_pet = batch[1]

(
    sub1_t1_pet.participant_id,
    sub1_t1_pet.image,
    sub2_t1_pet.participant_id,
    sub2_t1_pet.image,
)

# %%
