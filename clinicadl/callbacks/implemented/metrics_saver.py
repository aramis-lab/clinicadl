from __future__ import annotations

from typing import TYPE_CHECKING

from clinicadl.metrics.handler import MetricsHandler

from ..base import Callback

if TYPE_CHECKING:
    from clinicadl.io import Maps
    from clinicadl.train import TrainerState


class MetricsSaverCallback(Callback):
    """
    To save metrics in :py:class:`pd.DataFrame`.
    """

    def __init__(self):
        self._training_metrics = None
        self._checkpoint = None
        self._group_name = None

    def on_validation_end(self, *, metrics: MetricsHandler, **kwargs) -> None:
        self._training_metrics = metrics

    def on_train_end(
        self,
        *,
        maps: Maps,
        state: TrainerState,
        **kwargs,
    ) -> None:
        self._training_metrics.save(
            path=maps.training.splits[
                state.split_idx
            ].validation_metrics.aggregated_tsv,
            details_path=maps.training.splits[
                state.split_idx
            ].validation_metrics.details_tsv,
        )

    def on_validate_start(
        self,
        *,
        model_checkpoint: str,
        **kwargs,
    ) -> None:
        self._checkpoint = model_checkpoint

    def on_validate_end(
        self, *, maps: Maps, state: TrainerState, metrics: MetricsHandler, **kwargs
    ) -> None:
        chkpt_dir = maps.training.splits[state.split_idx].models.get_checkpoint_dir(
            self._checkpoint
        )
        metrics.merge(
            path=chkpt_dir.validation_metrics.aggregated_tsv,
            details_path=chkpt_dir.validation_metrics.details_tsv,
        )

    def on_test_start(
        self,
        *,
        model_checkpoint: str,
        group_name: str,
        **kwargs,
    ) -> None:
        self._checkpoint = model_checkpoint
        self._group_name = group_name

    def on_test_end(
        self,
        *,
        maps: Maps,
        metrics: MetricsHandler,
        **kwargs,
    ) -> None:
        split_idx, chkpt = maps.training.read_checkpoint_name(self._checkpoint)
        results_dir = (
            maps.test.groups[self._group_name].results.splits[split_idx].models[chkpt]
        )
        metrics.save(
            path=results_dir.metrics.aggregated_tsv,
            details_path=results_dir.metrics.details_tsv,
        )
