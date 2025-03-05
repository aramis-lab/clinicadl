from contextlib import nullcontext
from pathlib import Path

import pandas as pd
import torch
import torch.distributed as dist
import torchio.transforms as transforms
from monai.metrics.metric import Metric
from monai.metrics.regression import MAEMetric, MSEMetric
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from clinicadl.data.dataloader.config import DataLoaderConfig
from clinicadl.data.datasets.caps_dataset import CapsDataset
from clinicadl.data.datasets.concat import ConcatDataset
from clinicadl.data.datatype.preprocessing import PETLinear, T1Linear
from clinicadl.experiment_manager.maps_reader import MapsReader
from clinicadl.losses.config import CrossEntropyLossConfig, MSELossConfig
from clinicadl.losses.utils import Loss
from clinicadl.metrics.base import Metrics
from clinicadl.metrics.config.classification import (
    ConfusionMatrixMetricConfig,
    ROCAUCMetricConfig,
)
from clinicadl.metrics.factory import get_metric_from_config
from clinicadl.model.clinicadl_model import ClinicaDLModel
from clinicadl.networks.config.resnet import ResNet18Config, ResNetConfig
from clinicadl.networks.factory import ImplementedNetwork, get_network_config
from clinicadl.optim.optimizers.config import AdamConfig
from clinicadl.splitter import KFold, make_kfold, make_split
from clinicadl.trainer.trainer import Trainer
from clinicadl.transforms.extraction import Extraction, Image, Patch, Slice
from clinicadl.transforms.transforms import Transforms
from clinicadl.utils import cluster
from clinicadl.utils.early_stopping import EarlyStopping
from clinicadl.utils.seed import seed_everything

caps_directory = Path(
    "/Users/camille.brianceau/aramis/CLINICADL/caps"
)  # output of clinica pipelines
sub_ses_t1 = Path(
    "/Users/camille.brianceau/aramis/CLINICADL/caps/subjects_t1.tsv"
)  # 64 subjects

preprocessing_t1 = T1Linear()
transforms_image = Transforms(
    extraction=Slice(slices=[24, 56, 78]),
)

dataset_t1_image = CapsDataset(
    caps_directory=caps_directory,
    data=sub_ses_t1,
    preprocessing=preprocessing_t1,
    transforms=transforms_image,
    label="diagnosis",  # need to have a "diagnosis" column in the tsv/df given (data)
)
dataset_t1_image.to_tensors(n_proc=2)  # to extract the tensor of the T1 file

print("done")

split_dir = make_split(sub_ses_t1, n_test=0.2)  # Optional data tsv and output_dir
fold_dir = make_kfold(split_dir / "train.tsv", n_splits=2)
splitter = KFold(fold_dir)  # train : 24,  train baseline : 16 , val baseline : 16


maps_path = Path("maps_test")
maps_reader = MapsReader(maps_path)  #

config_file = Path("config_file")
trainer = Trainer(maps_path)

### PARAMETERS #####
seed = 3
batch_size: int = 2
epochs: int = 3
lr: float = 0.1
weight_decay: float = 0.0
momentum: float = 0.9
num_workers: int = 0
persistent_workers: bool = True
pin_memory: bool = True

non_blocking: bool = True
prefetch_factor: int = 0
drop_last: bool = True
amp: bool = True
accumulation_steps: int = 1  # gives the number of iterations during which gradients are accumulated before performing the weights update. This allows to virtually increase the size of the batch. Default: 1.
evaluation_steps: int = 5  # gives the number of iterations to perform an evaluation internal to an epoch. Default will only perform an evaluation at the end of each epoch.
current_epoch = 0
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
tolerance = 0
patience = 10
num_replica = cluster.size
mini_batch_size = batch_size
global_batch_size = mini_batch_size * num_replica


seed_everything(seed, deterministic=False, compensation="memory")
###########################


dataloader_config = DataLoaderConfig(
    batch_size=mini_batch_size,
    sampling_weights=None,
    shuffle=False,
    drop_last=drop_last,
    num_workers=num_workers,
    prefetch_factor=None,
    pin_memory=pin_memory,
)  #  persistent_workers=self.persistent_workers,

for split in splitter.get_splits(dataset=dataset_t1_image):
    model = ClinicaDLModel.from_config(
        network_config=get_network_config(
            ImplementedNetwork.RESNET, num_outputs=3, spatial_dims=2, in_channels=1
        ),
        loss_config=MSELossConfig(),
        optimizer_config=AdamConfig(),
    )

    print(f"Training split {split}")

    #########  DATALOADER ############
    print("Loading training data...")
    # TODO: Deal with sampler and multi GPU and data //
    split.build_train_loader(dataloader_config)

    print(split.train_loader.type)
    split.build_val_loader(dataloader_config)
    ###########################

    print("########## TRAIN BEGIN ############ ")
    # TODO: maybe we can put all of that in a callnacks/function on_train_begin()

    # Initialisation du modèle et des paramètres
    model.network.to(device)
    model.network.train()

    epoch = current_epoch  # will be different if resume or transfer learning
    early_stopping = EarlyStopping("min", min_delta=tolerance, patience=patience)
    scaler = GradScaler(enabled=amp)
    # profiler = init_profiler(maps_path)

    # TODO: init tracker like WandB or MlFlow (callbacks ?)

    # dataset size 8 x nb de slice
    N_batch = len(split.train_loader)

    # Vérification de evaluation_steps
    if evaluation_steps >= N_batch:
        print(
            f"Warning: evaluation_steps ({evaluation_steps}) >= N_batch ({N_batch}) ! Réduction automatique à N_batch // 2."
        )
        evaluation_steps = max(1, N_batch // 2)  # Évite d'avoir une valeur trop grande

    elif N_batch % evaluation_steps != 0:
        print(
            f"Warning: evaluation_steps ({evaluation_steps}) ne divise pas exactement N_batch ({N_batch})."
        )
        evaluation_steps = max(
            1, min(evaluation_steps, N_batch // 2)
        )  # Ajuste pour garder une fréquence raisonnable

    N_val_batch = len(split.val_loader)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        model.optimizer, max_lr=lr, steps_per_epoch=N_batch, epochs=epochs
    )

    # INIT METRICS
    metrics = Metrics([MAEMetric(), MSEMetric()], loss=model.loss)

    while epoch < epochs and not early_stopping.step(metrics.val_loss):
        ############## ON EPOCH BEGIN #################

        print(f"########## EPOCH {epoch} BEGIN ############ ")

        if isinstance(split.train_loader.sampler, DistributedSampler):
            # It should always be true for a random sampler. But just in case
            # we get a WeightedRandomSampler or a forgotten RandomSampler,
            # we do not want to execute this line.
            split.train_loader.sampler.set_epoch(epoch)

        model.network.zero_grad(set_to_none=True)
        evaluation_flag, step_flag = True, True

        for i, data in enumerate(split.train_loader):
            ################# BATCH BEGIN #################
            print(f"########## BATCH {i} BEGIN ############ ")

            # TODO: to remove and to put in the dataloader
            images = (
                torch.cat(list(i.sample for i in data), dim=0).unsqueeze(1).to(device)
            )
            print(images.shape)  # Should be (batch_size, channels, height, width)

            labels = (
                torch.tensor([i.label for i in data], dtype=torch.float32)
                .unsqueeze(1)
                .to(device)
            )  # TO REMOVE AND CHECK FOR MASK
            ############

            with autocast(device.type, enabled=amp):
                outputs = model.network(images)
                loss = model.loss(outputs, labels) / accumulation_steps
                # metrics.df_train[(i, epoch), f"{MSEMetric().__str__()}"] = MSEMetric()(outputs, labels)
                # metrics.df_train[(i, epoch), f"{model.loss.__str__()}"] = loss
                print(loss)
            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                step_flag = False
                scaler.step(model.optimizer)
                scaler.update()
                model.optimizer.zero_grad(set_to_none=True)

                del loss

                # Evaluate the model only when no gradients are accumulated
                if evaluation_steps != 0 and (i + 1) % evaluation_steps == 0:
                    evaluation_flag = False
                    print(" Evaluate the model only when no gradients are accumulated")
                    print(f"Évaluation - Epoch {epoch}, Batch {i}:")
                    test(split.val_loader, model, device, amp)

        # PROFILER STEP

        # If no step has been performed, raise Exception
        if step_flag:
            from clinicadl.utils.exceptions import ClinicaDLTrainingException

            raise ClinicaDLTrainingException(
                "The model has not been updated once in the epoch. The accumulation step may be too large."
            )

        # If no evaluation has been performed, warn the user
        elif evaluation_flag and evaluation_steps != 0:
            print(
                f"Your evaluation steps {evaluation_steps} are too big "
                f"compared to the size of the dataset. "
                f"The model is evaluated only once at the end epochs."
            )

        # Update weights one last time if gradients were computed without update
        if (i + 1) % accumulation_steps != 0:
            scaler.step(model.optimizer)
            scaler.update()
            model.optimizer.zero_grad(set_to_none=True)

        # Always test the results and save them once at the end of the epoch
        model.network.zero_grad(set_to_none=True)
        print(f"Last checkpoint at the end of the epoch {epoch}")
        print("training data")
        test(split.train_loader, model, device, amp)

        print("validation data")
        test(split.val_loader, model, device, amp)
        print(metrics.df_train)

        ############ ON EPOCH END ##########

        # model_weights = {
        #     "model": model.load_state_dict(),
        #     "epoch": epoch,
        #     "name": model.network.__str__(), #???
        # }
        # optimizer_weights = {
        #     "optimizer": model.load_optim_state_dict(model.optimizer),
        #     "epoch": epoch,
        #     "name": model.network.__str__(), #???
        # }

        # if cluster.master:
        #     write optimizer and model weight
        # dist.barrier()

        ###### ON STEP END
        #  ########
        scheduler.step()  # Update learning rate based on validation loss

        print("increase epoch")
        epoch += 1
        # Sauvegarde du modèle à la fin de chaque epoch
        torch.save(
            model.network.state_dict(), maps_path / f"model_epoch_{epoch}.pth"
        )  # model.save_checkpoint(epoch = epoch)

    # del model.network and then reload model weights

    # Last
    print("training data")
    # scale ?
    test(split.train_loader, model, device, amp)

    print("validation data")
    test(split.val_loader, model, device, amp)

    ########## ON TRAIN END ###########

    # if cluster.master:
    #     ensemble_prediction(
    #         "train",
    #         split,
    #         selection_metrics,
    #     )
    #     ensemble_prediction(
    #         "validation",
    #         split,
    #         selection_metrics,
    #     )

    #     erase_tmp(split)
