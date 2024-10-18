# %% class
class MapsIO:
    pass


class CapsDataset:
    pass


class Splitter:
    pass


class ClinicaDLModels:
    pass


class Networks:
    pass


class VAE(Networks):
    pass


class Optimizer:
    pass


class Loss:
    pass


class Metrics:
    pass


class Trainer:
    pass


class Validator:
    pass


# %% maps
maps = MapsIO("/path/to/maps")  # Crée un dossier

# %% Dataset
DataConfig = {
    "caps_dir": "",
    "tsv": "",
    "mode": "",
}
capsdataset = CapsDataset(DataConfig, maps)  # torch.dataset

# %% Model
network = VAE()  # nn.module
loss = Loss()
optimizer = ClinicaDLOptim(
    Adam()
)  # get_optimizer({"name": "Adam", "par1": 0.5}) # torch.optim
# model = ClinicaDLModels(
#     network,
#     loss,
#     optimizer,
# )

# %% Cross val
SplitConfig = SplitterConfig()
splitter = Splitter(SplitConfig, capsdataset)

# %% Metrics
metrics1 = Metrics("MAE")  # monai.metric
metrics2 = Metrics("MSE")  # monai.metric


# %% Option 1
for split in splitter.iterate():
    trainer = Trainer(split, maps, (optimizer))
    validator = Validator(split, [metrics1, metrics2], maps)

    trainer.train(validator, model)


# %% Option 2
val = Validator([metrics1, metrics2], maps)
trainer = Trainer(validator, maps)
for split in splitter.iterate():
    trainer.train(model, split)

# %% Option 3
trainer = Trainer(
    maps, [metrics1, metrics2]
)  # Initialise un maps manager + initialise un validator
for split in splitter.iterate():
    model = ClinicaDLModels(
        network,
        loss,
        optimizer,
    )
    trainer.train(model, split)
