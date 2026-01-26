from unittest.mock import MagicMock

from clinicadl.train import TrainerState

DATALOADER = MagicMock()


def test_trainer_state():
    state = TrainerState()
    state.stage = "training"
    state.called = "train"
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
        "called": "train",
        "stage": "training",
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
    DATALOADER.__len__.return_value = 2
    state.reset_prediction(dataloader=DATALOADER)
    assert state.current_pred_batch == 0
    assert state.num_pred_batches == 2
    assert state.split_idx is None
    assert state.stage == "prediction"
    assert state.called == "predict"

    state.reset_validation(split_idx=1, val_loader=DATALOADER, in_training=False)
    assert state.current_val_batch == 0
    assert state.num_val_batches == 2
    assert state.split_idx == 1
    assert state.stage == "evaluation"
    assert state.called == "validate"

    DATALOADER.__len__.return_value = 3
    state.reset_training(split_idx=2, num_epochs=5)
    assert state.current_train_batch == 0
    assert state.num_train_batches == 0
    assert state.current_val_batch == 0
    assert state.num_val_batches == 0
    assert state.split_idx == 2
    assert state.called == "train"
    assert not state.should_stop
    assert state.current_epoch == 0
    assert state.num_epochs == 5
    assert state.optim_step == 0

    state.reset_validation(split_idx=1, val_loader=DATALOADER, in_training=True)
    assert state.called == "train"
    assert state.stage == "evaluation"

    state.current_train_batch = 2
    state.num_train_batches = 5
    state.optim_step = 2
    DATALOADER.__len__.return_value = 3
    state.reset_epoch(train_loader=DATALOADER, current_epoch=7)
    assert state.stage == "training"
    assert state.current_epoch == 7
    assert state.current_train_batch == 0
    assert state.num_train_batches == 3
    assert state.optim_step == 0

    state.reset_test(dataloader=DATALOADER)
    assert state.current_test_batch == 0
    assert state.num_test_batches == 3
    assert state.split_idx is None
    assert state.stage == "evaluation"
    assert state.called == "test"
    assert state.current_train_batch == 0

    state.load_state_dict(state_dict)
    assert state.stage == "training"
    assert state.called == "train"
    assert state.current_train_batch == 1
    assert state.current_pred_batch == 1
    assert state.current_val_batch == 1
    assert state.current_test_batch == 1
    assert state.current_epoch == 1
    assert state.optim_step == 1
    assert state.should_stop
    assert state.split_idx == 1
