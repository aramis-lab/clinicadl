from dataclasses import dataclass

from clinicadl.train import TrainerState


@dataclass
class DataLoader:
    len_: int = 2

    def __len__(self):
        return self.len_


@dataclass
class Split:
    def __init__(self, len_: int = 2, index: int = 1):
        self.index = index
        self.train_loader = DataLoader(len_=len_)
        self.val_loader = DataLoader(len_=len_)


def test_trainer_state():
    state = TrainerState()
    state.stage = "test"
    state.current_train_batch = 1
    state.current_pred_batch = 1
    state.current_val_batch = 1
    state.current_test_batch = 1
    state.current_epoch = 1
    state.optim_step = 1
    state.should_stop = True
    state.split_idx = 1

    state_dict = state.state_dict()
    assert state_dict == {
        "stage": "test",
        "should_stop": True,
        "current_train_batch": 1,
        "num_train_batches": 0,
        "current_val_batch": 1,
        "num_val_batches": 0,
        "current_pred_batch": 1,
        "num_pred_batches": 0,
        "current_test_batch": 1,
        "num_test_batches": 0,
        "current_epoch": 1,
        "num_epochs": 0,
        "optim_step": 1,
        "split_idx": 1,
    }
    state.reset_prediction(dataloader=DataLoader())
    assert state.current_pred_batch == 0
    assert state.num_pred_batches == 2
    assert state.split_idx is None
    assert state.stage == "prediction"
    assert state.current_train_batch == 1

    state.reset_validation(split=Split())
    assert state.current_val_batch == 0
    assert state.num_val_batches == 2
    assert state.split_idx == 1
    assert state.stage == "validation"
    assert state.current_train_batch == 1

    state.reset_test(dataloader=DataLoader())
    assert state.current_test_batch == 0
    assert state.num_test_batches == 2
    assert state.split_idx is None
    assert state.stage == "test"
    assert state.current_train_batch == 1

    state.reset_train(split=Split(index=2, len_=3), num_epochs=5)
    assert state.current_train_batch == 0
    assert state.num_train_batches == 3
    assert state.current_val_batch == 0
    assert state.num_val_batches == 3
    assert state.split_idx == 2
    assert state.stage == "training"
    assert not state.should_stop
    assert state.current_epoch == 0
    assert state.num_epochs == 5
    assert state.optim_step == 0

    state.load_state_dict(state_dict)
    assert state.stage == "test"
    assert state.current_train_batch == 1
    assert state.current_pred_batch == 1
    assert state.current_val_batch == 1
    assert state.current_test_batch == 1
    assert state.current_epoch == 1
    assert state.optim_step == 1
    assert state.should_stop
    assert state.split_idx == 1
