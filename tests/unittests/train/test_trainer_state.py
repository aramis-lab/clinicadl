from clinicadl.train import TrainerState


def test_trainer_state():
    state = TrainerState()
    state.current_train_batch = 1
    state.current_pred_batch = 1
    state.current_val_batch = 1
    state.current_epoch = 1
    state.optim_step = 1
    state.should_stop = True
    state.split_idx = 1

    state_dict = state.state_dict()
    assert state_dict == {
        "should_stop": True,
        "current_train_batch": 1,
        "num_train_batches": 0,
        "current_val_batch": 1,
        "num_val_batches": 0,
        "current_pred_batch": 1,
        "num_pred_batches": 0,
        "current_epoch": 1,
        "num_epochs": 0,
        "optim_step": 1,
        "split_idx": 1,
    }
    state.reset_prediction()
    assert state.num_pred_batches == 0
    assert state.num_train_batches == 1
    state.reset_validation()
    assert state.num_val_batches == 0
    assert state.num_train_batches == 1
    state.reset()
    assert state.current_train_batch == 0
    assert state.current_pred_batch == 0
    assert state.current_val_batch == 0
    assert state.current_epoch == 0
    assert state.optim_step == 0
    assert not state.should_stop
    assert state.split_idx is None

    state.load_state_dict(state_dict)
    assert state.current_train_batch == 1
    assert state.current_pred_batch == 1
    assert state.current_val_batch == 1
    assert state.current_epoch == 1
    assert state.optim_step == 1
    assert state.should_stop
    assert state.split_idx == 1
