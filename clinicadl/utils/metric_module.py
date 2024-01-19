from logging import getLogger
from typing import Dict, List
from sklearn.utils import resample

import numpy as np

metric_optimum = {
    "MAE": "min",
    "MSE": "min",
    "accuracy": "max",
    "sensitivity": "max",
    "specificity": "max",
    "PPV": "max",
    "NPV": "max",
    "F1_score": "max",
    "BA": "max",
    "PSNR": "max",
    "SSIM": "max",
    "LNCC": "max",
    "loss": "min",
}

logger = getLogger("clinicadl.metric")


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
                raise NotImplementedError(
                    f"The metric {metric} is not implemented in the module."
                )


    def compute_confidence_interval(self,
                                y, 
                                y_pred, 
                                metric_fn, 
                                class_number=None, 
                                confidence_level=0.95, 
                                num_bootstrap_samples=1000):

    
        """
        Compute confidence interval for a given metric using bootstrapping.

        Args:
            y (array-like): True labels.
            y_pred (array-like): Predicted labels.
            metric_fn (callable): Metric function.
            class_number (int, optional): Class number for class-specific metrics.
            confidence_level (float, optional): Desired confidence level for intervals.
            num_bootstrap_samples (int, optional): Number of bootstrap samples.

        Returns:
            Tuple[float, float, float]: Lower bound, upper bound, and standard error of the metric.
        """
        bootstrap_samples = np.zeros(num_bootstrap_samples)

        for i in range(num_bootstrap_samples):
            indices = np.random.choice(len(y), len(y), replace=True)
            y_bootstrap, y_pred_bootstrap = y[indices], y_pred[indices]

            if class_number is not None:
                bootstrap_samples[i] = metric_fn(y_bootstrap, y_pred_bootstrap, class_number)
            else:
                bootstrap_samples[i] = metric_fn(y_bootstrap, y_pred_bootstrap)

        lower_ci, upper_ci = np.percentile(bootstrap_samples, 
                                        [(1 - confidence_level) / 2 * 100, (1 + confidence_level) / 2 * 100])
        
        standard_error = np.std(bootstrap_samples)

        return lower_ci, upper_ci, standard_error

    def apply(self, y, y_pred, ci):
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

            metric_names = ["Metrics"]
            metric_values = ["Values"]  # Collect metric values
            lower_ci_values = ["Lower bound CI"]  # Collect lower CI values
            upper_ci_values = ["Upper bound CI"]  # Collect upper CI values
            se_values = ["SE"]  # Collect standard error values
            for metric_key, metric_fn in self.metrics.items():
                
                metric_args = list(metric_fn.__code__.co_varnames)
                if "class_number" in metric_args and self.n_classes > 2:
                    for class_number in range(self.n_classes):
                        if ci : 
                            metric_result = metric_fn(y, y_pred, class_number)
                            lower_ci, upper_ci, standard_error = self.compute_confidence_interval(y, y_pred, metric_fn, class_number)

                            metric_values.append(metric_result)
                            lower_ci_values.append(lower_ci)
                            upper_ci_values.append(upper_ci)
                            se_values.append(standard_error)
                            metric_names.append(f"{metric_key}-{class_number}")
                        else: 
                            results[f"{metric_key}-{class_number}"] = metric_fn(
                            y, y_pred, class_number
                        )

                elif "class_number" in metric_args:
                    if ci:
                        metric_result = metric_fn(y, y_pred, 0)
                        metric_values.append(metric_result)
                        lower_ci, upper_ci, standard_error = self.compute_confidence_interval(y, y_pred, metric_fn, 0)
                        lower_ci_values.append(lower_ci)
                        upper_ci_values.append(upper_ci)
                        se_values.append(standard_error)
                        metric_names.append(f"{metric_key}")
                    else:
                        results[f"{metric_key}"] = metric_fn(y, y_pred, 0)

                else:
                    if ci:
                        metric_result = metric_fn(y, y_pred)
                        metric_values.append(metric_result)
                        lower_ci, upper_ci, standard_error = self.compute_confidence_interval(y, y_pred, metric_fn)
                        lower_ci_values.append(lower_ci)
                        upper_ci_values.append(upper_ci)
                        se_values.append(standard_error)
                        metric_names.append(f"{metric_key}")
                    else:
                        results[f"{metric_key}"] = metric_fn(y, y_pred)

            if ci:
                # Construct the final results dictionary
                results["Metric_names"] = metric_names
                results["Metric_values"] = metric_values
                results["Lower_CI"] = lower_ci_values
                results["Upper_CI"] = upper_ci_values
                results["SE"] = se_values

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
    def f1_score_fn(y, y_pred, class_number):
        """
        Args:
            y (List): list of labels
            y_pred (List): list of predictions
            class_number (int): number of the class studied
        Returns:
            (float) F1 score
        """
        
        precision = MetricModule.ppv_fn(y, y_pred, class_number)
        recall = MetricModule.sensitivity_fn(y, y_pred, class_number)

        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

        return f1_score

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

    @staticmethod
    def ssim_fn(y, y_pred):
        """
        Args:
            y (List): list of labels
            y_pred (List): list of predictions
        Returns:
            (float) SSIM
        """
        from clinicadl.utils.pytorch_ssim import ssim, ssim3D

        if len(y) == 3:
            return ssim(y, y_pred).item()
        else:
            return ssim3D(y, y_pred).item()

    @staticmethod
    def psnr_fn(y, y_pred):
        """
        Args:
            y (List): list of labels
            y_pred (List): list of predictions
        Returns:
            (float) PSNR
        """
        from skimage.metrics import peak_signal_noise_ratio

        return peak_signal_noise_ratio(y, y_pred)

    @staticmethod
    def lncc_fn(y, y_pred):
        """
        Args:
            y (List): list of labels
            y_pred (List): list of predictions
        Returns:
            (float) LNCC
        """
        from scipy.ndimage import gaussian_filter

        sigma = 2

        mean1 = gaussian_filter(y, sigma)
        mean2 = gaussian_filter(y_pred, sigma)

        mean12 = gaussian_filter(
            y * y_pred, sigma
        )  # the * operator is term by term product
        mean11 = gaussian_filter(y * y, sigma)
        mean22 = gaussian_filter(y_pred * y_pred, sigma)

        covar12 = mean12 - (mean1 * mean2)
        var1 = np.sqrt(mean11 - (mean1 * mean1))
        var2 = np.sqrt(mean22 - (mean2 * mean2))

        lcc_matrix = np.maximum(covar12 / (var1 * var2), 0)
        return np.mean(lcc_matrix)


class RetainBest:
    """
    A class to retain the best and overfitting values for a set of wanted metrics.
    """

    def __init__(self, selection_metrics: List[str], n_classes: int = 0):
        self.selection_metrics = selection_metrics

        if "loss" in selection_metrics:
            selection_metrics.remove("loss")
            metric_module = MetricModule(selection_metrics)
            selection_metrics.append("loss")
        else:
            metric_module = MetricModule(selection_metrics)

        implemented_metrics = set(metric_optimum.keys())
        if not set(self.selection_metrics).issubset(implemented_metrics):
            raise NotImplementedError(
                f"The selection metrics {self.selection_metrics} are not all implemented. "
                f"Available metrics are {implemented_metrics}."
            )
        self.best_metrics = dict()
        for selection in self.selection_metrics:
            if n_classes > 2:
                metric_fn = metric_module.metrics[selection]
                metric_args = list(metric_fn.__code__.co_varnames)
                if "class_number" in metric_args:
                    for class_number in range(n_classes):
                        self.set_optimum(f"{selection}-{class_number}")
                else:
                    self.set_optimum(selection)
            else:
                self.set_optimum(selection)

    def set_optimum(self, selection: str):
        if metric_optimum[selection] == "min":
            self.best_metrics[selection] = np.inf
        elif metric_optimum[selection] == "max":
            self.best_metrics[selection] = -np.inf
        else:
            raise ValueError(
                f"Objective {metric_optimum[selection]} unknown for metric {selection}."
                f"Please choose between 'min' and 'max'."
            )

    def step(self, metrics_valid: Dict[str, float]) -> Dict[str, bool]:
        """
        Computes for each metric if this is the best value ever seen.

        Args:
            metrics_valid: metrics computed on the validation set
        Returns:
            metric is associated to True if it is the best value ever seen.
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
