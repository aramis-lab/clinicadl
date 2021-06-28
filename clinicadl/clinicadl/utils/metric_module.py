import numpy as np

metric_optimum = {
    "mae": "min",
    "accuracy": "max",
    "sensitivity": "max",
    "specificity": "max",
    "ppv": "max",
    "npv": "max",
    "ba": "max",
    "loss": "min",
}

# TODO: what about the loss?


class MetricModule:
    def __init__(self, metrics):

        # Check if wanted metrics are implemented
        list_fn = [
            method_name
            for method_name in dir(MetricModule)
            if callable(getattr(MetricModule, method_name))
        ]
        self.metrics = dict()
        for metric in metrics:
            if f"{metric.lower()}_fn" in list_fn:
                self.metrics[metric] = getattr(MetricModule, f"{metric}_fn")
            else:
                raise ValueError(
                    f"The metric {metric} is not implemented in the module"
                )

    def apply(self, y, y_pred):
        """
        This is a function to calculate the different metrics based on the list of true label and predicted label

        Args:
            y (List): list of labels
            y_pred (List): list of predictions
        Returns:
            (Dict[str:float]) metrics results
        """

        if y is not None and y_pred is not None:
            results = dict()
            y = np.array(y)
            y_pred = np.array(y_pred)

            for metric_key, metric_fn in self.metrics.items():
                results[metric_key] = metric_fn(y, y_pred)
        else:
            results = dict()

        return results

    @staticmethod
    def mae_fn(y, y_pred):
        """
        Args:
            y (List): list of labels
            y_pred (List): list of predictions
        Returns:
            (float) mean absolute error
        """

        return np.mean(np.abs(y - y_pred))

    @staticmethod
    def accuracy_fn(y, y_pred):
        """
        Args:
            y (List): list of labels
            y_pred (List): list of predictions
        Returns:
            (float) accuracy
        """
        true_positive = np.sum((y_pred == 1) & (y == 1))
        true_negative = np.sum((y_pred == 0) & (y == 0))
        false_positive = np.sum((y_pred == 1) & (y == 0))
        false_negative = np.sum((y_pred == 0) & (y == 1))

        return (true_positive + true_negative) / (
            true_positive + true_negative + false_positive + false_negative
        )

    @staticmethod
    def sensitivity_fn(y, y_pred):
        """
        Args:
            y (List): list of labels
            y_pred (List): list of predictions
        Returns:
            (float) sensitivity
        """
        true_positive = np.sum((y_pred == 1) & (y == 1))
        false_negative = np.sum((y_pred == 0) & (y == 1))

        if (true_positive + false_negative) != 0:
            return true_positive / (true_positive + false_negative)
        else:
            return 0.0

    @staticmethod
    def specificity_fn(y, y_pred):
        """
        Args:
            y (List): list of labels
            y_pred (List): list of predictions
        Returns:
            (float) specificity
        """
        true_negative = np.sum((y_pred == 0) & (y == 0))
        false_positive = np.sum((y_pred == 1) & (y == 0))

        if (false_positive + true_negative) != 0:
            return true_negative / (false_positive + true_negative)
        else:
            return 0.0

    @staticmethod
    def ppv_fn(y, y_pred):
        """
        Args:
            y (List): list of labels
            y_pred (List): list of predictions
        Returns:
            (float) positive predictive value
        """
        true_positive = np.sum((y_pred == 1) & (y == 1))
        false_positive = np.sum((y_pred == 1) & (y == 0))

        if (true_positive + false_positive) != 0:
            return true_positive / (true_positive + false_positive)
        else:
            return 0.0

    @staticmethod
    def npv_fn(y, y_pred):
        """
        Args:
            y (List): list of labels
            y_pred (List): list of predictions
        Returns:
            (float) negative predictive value
        """
        true_negative = np.sum((y_pred == 0) & (y == 0))
        false_negative = np.sum((y_pred == 0) & (y == 1))

        if (true_negative + false_negative) != 0:
            return true_negative / (true_negative + false_negative)
        else:
            return 0.0

    @staticmethod
    def ba_fn(y, y_pred):
        """
        Args:
            y (List): list of labels
            y_pred (List): list of predictions
        Returns:
            (float) balanced accuracy
        """

        return (
            MetricModule.sensitivity_fn(y, y_pred)
            + MetricModule.specificity_fn(y, y_pred)
        ) / 2

    @staticmethod
    def confusion_matrix_fn(y, y_pred):
        """
        Args:
            y (List): list of labels
            y_pred (List): list of predictions
        Returns:
            (Dict[str:float]) confusion matrix
        """
        true_positive = np.sum((y_pred == 1) & (y == 1))
        true_negative = np.sum((y_pred == 0) & (y == 0))
        false_positive = np.sum((y_pred == 1) & (y == 0))
        false_negative = np.sum((y_pred == 0) & (y == 1))

        return {
            "tp": true_positive,
            "tn": true_negative,
            "fp": false_positive,
            "fn": false_negative,
        }


class RetainBest:
    """
    A class to retain the best and overfitting values for a set of wanted metrics.
    """

    def __init__(self, selection_list):
        self.selection_list = selection_list
        self._init_best_metrics()

    def _init_best_metrics(self):
        self.best_metrics = dict()
        for selection in self.selection_list:
            if metric_optimum[selection.lower()] == "min":
                self.best_metrics[selection] = np.inf
            elif metric_optimum[selection.lower()] == "max":
                self.best_metrics[selection] = -np.inf
            else:
                raise ValueError(
                    f"Objective {objective} unknown for metric {name}."
                    f"Please choose between 'min' and 'max'."
                )

    def step(self, metrics_valid):
        """
        Returns a dictionnary of boolean. A metric is associated to True if it the best value ever seen.

        Args:
            metrics_valid: (Dict[str:int]) metrics computed on the validation set
        Returns:
            (Dict[str:bool])
        """

        metrics_dict = dict()
        for selection in self.selection_list:
            if metric_optimum[selection.lower()] == "min":
                metrics_dict[selection] = (
                    metrics_valid[selection] < self.best_metrics[selection]
                )
                self.best_metrics[selection] = min(
                    metrics_valid[selection], self.best_metrics[selection]
                )

            else:
                metrics_dict[selection] = (
                    metrics_valid[selection] > self.best_metrics[selection]
                )
                self.best_metrics[selection] = max(
                    metrics_valid[selection], self.best_metrics[selection]
                )

        return metrics_dict
