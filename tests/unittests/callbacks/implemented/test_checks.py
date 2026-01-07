import re
from unittest.mock import Mock

import pytest
import torch

from clinicadl.callbacks.implemented import ChecksCallback

MODEL = Mock()
LOSS = Mock()


class TestCheckLosses:
    checker = ChecksCallback()

    def test_on_train_start(self):
        MODEL.get_loss_functions.return_value = {}
        with pytest.raises(
            ValueError,
            match="clinicadl.models.Model.get_loss_functions method should return a dictionary with at least one key.",
        ):
            self.checker.on_train_start(model=MODEL)
        MODEL.get_loss_functions.return_value = {"my_loss": LOSS}
        self.checker.on_train_start(model=MODEL)

    def test_on_backward_step_start(self):
        MODEL.get_loss_functions.return_value = {"my_loss": LOSS, "other_loss": LOSS}
        self.checker.on_train_start(model=MODEL)

        with pytest.raises(
            ValueError,
            match="clinicadl.models.Model.forward_step should return a Tensor, or a dict of Tensors. Got:.*",
        ):
            self.checker.on_backward_step_start(loss=1.1)
        with pytest.raises(
            ValueError,
            match="clinicadl.models.Model.forward_step should return a Tensor, or a dict of Tensors. Got:.*",
        ):
            self.checker.on_backward_step_start(loss={"my_loss": 1.1})
        with pytest.raises(
            ValueError,
            match=re.escape(
                "clinicadl.models.Model.forward_step returns a single loss, whereas clinicadl.models.Model.get_loss_functions returns 2 loss function(s) ['my_loss', 'other_loss']"
            ),
        ):
            self.checker.on_backward_step_start(loss=torch.tensor([1.1]))
        with pytest.raises(
            ValueError,
            match=re.escape(
                "clinicadl.models.Model.forward_step returns loss(es) named ['my_loss'], whereas clinicadl.models.Model.get_loss_functions "
                "returns ['my_loss', 'other_loss'] loss function(s)"
            ),
        ):
            self.checker.on_backward_step_start(loss={"my_loss": torch.tensor([1.1])})

        self.checker.on_backward_step_start(
            loss={"my_loss": torch.tensor([1.1]), "other_loss": torch.tensor([1.2])},
        )
        assert self.checker._check_losses._checked
        self.checker.on_backward_step_start(
            loss={"my_loss": torch.tensor([1.1]), "other_loss": torch.tensor([1.2])},
        )
        assert self.checker._check_losses._checked

        MODEL.get_loss_functions.return_value = {"my_loss": LOSS}
        self.checker.on_train_start(model=MODEL)
        self.checker.on_backward_step_start(loss=torch.tensor([1]))
