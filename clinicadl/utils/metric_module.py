import numpy as np

metric_optimum = {
    "MAE": "min",
    "MSE": "min",
    "accuracy": "max",
    "sensitivity": "max",
    "specificity": "max",
    "PPV": "max",
    "NPV": "max",
    "BA": "max",
    "loss": "min",
}


class MetricModule:
    def __init__(self, metrics, n_classes=2):

        self.n_classes = n_classes

        # Check if wanted metrics are implemented
        list_fn = [
            method_name
            for method_name in dir(MetricModule)
            if callable(getattr(MetricModule, method_name))
        ]
        self.metrics = dict()
        for metric in metrics:
            if f"{metric.lower()}_fn" in list_fn:
                self.metrics[metric] = getattr(MetricModule, f"{metric.lower()}_fn")
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
                metric_args = list(metric_fn.__code__.co_varnames)
                if "class_number" in metric_args:
                    for class_number in range(self.n_classes):
                        results[f"{metric_key}-{class_number}"] = metric_fn(
                            y, y_pred, class_number
                        )
                else:
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
    def mse_fn(y, y_pred):
        """
        Args:
            y (List): list of labels
            y_pred (List): list of predictions
        Returns:
            (float) mean squared error
        """

        return np.mean(np.square(y - y_pred))

    @staticmethod
    def accuracy_fn(y, y_pred):
        """
        Args:
            y (List): list of labels
            y_pred (List): list of predictions
        Returns:
            (float) accuracy
        """
        true = np.sum(y_pred == y)

        return true / len(y)

    @staticmethod
    def sensitivity_fn(y, y_pred, class_number):
        """
        Args:
            y (List): list of labels
            y_pred (List): list of predictions
            class_number (int): number of the class studied
        Returns:
            (float) sensitivity
        """
        true_positive = np.sum((y_pred == class_number) & (y == class_number))
        false_negative = np.sum((y_pred != class_number) & (y == class_number))

        if (true_positive + false_negative) != 0:
            return true_positive / (true_positive + false_negative)
        else:
            return 0.0

    @staticmethod
    def specificity_fn(y, y_pred, class_number):
        """
        Args:
            y (List): list of labels
            y_pred (List): list of predictions
            class_number (int): number of the class studied
        Returns:
            (float) specificity
        """
        true_negative = np.sum((y_pred != class_number) & (y != class_number))
        false_positive = np.sum((y_pred == class_number) & (y != class_number))

        if (false_positive + true_negative) != 0:
            return true_negative / (false_positive + true_negative)
        else:
            return 0.0

    @staticmethod
    def ppv_fn(y, y_pred, class_number):
        """
        Args:
            y (List): list of labels
            y_pred (List): list of predictions
            class_number (int): number of the class studied
        Returns:
            (float) positive predictive value
        """
        true_positive = np.sum((y_pred == class_number) & (y == class_number))
        false_positive = np.sum((y_pred == class_number) & (y != class_number))

        if (true_positive + false_positive) != 0:
            return true_positive / (true_positive + false_positive)
        else:
            return 0.0

    @staticmethod
    def npv_fn(y, y_pred, class_number):
        """
        Args:
            y (List): list of labels
            y_pred (List): list of predictions
            class_number (int): number of the class studied
        Returns:
            (float) negative predictive value
        """
        true_negative = np.sum((y_pred != class_number) & (y != class_number))
        false_negative = np.sum((y_pred != class_number) & (y == class_number))

        if (true_negative + false_negative) != 0:
            return true_negative / (true_negative + false_negative)
        else:
            return 0.0

    @staticmethod
    def ba_fn(y, y_pred, class_number):
        """
        Args:
            y (List): list of labels
            y_pred (List): list of predictions
            class_number (int): number of the class studied
        Returns:
            (float) balanced accuracy
        """

        return (
            MetricModule.sensitivity_fn(y, y_pred, class_number)
            + MetricModule.specificity_fn(y, y_pred, class_number)
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

    def __init__(self, selection_metrics):
        self.selection_metrics = selection_metrics
        implemented_metrics = set(metric_optimum.keys())
        if not set(selection_metrics).issubset(implemented_metrics):
            raise NotImplementedError(
                f"The selection metrics {selection_metrics} are not all implemented. "
                f"Available metrics are {implemented_metrics}."
            )
        self.best_metrics = dict()
        for selection in self.selection_metrics:
            if metric_optimum[selection] == "min":
                self.best_metrics[selection] = np.inf
            elif metric_optimum[selection] == "max":
                self.best_metrics[selection] = -np.inf
            else:
                raise ValueError(
                    f"Objective {metric_optimum[selection]} unknown for metric {selection}."
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
        for selection in self.selection_metrics:
            if metric_optimum[selection] == "min":
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
