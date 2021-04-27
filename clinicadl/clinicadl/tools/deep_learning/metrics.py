import numpy as np


class MetricModule:
    def __init__(self, metrics):

        # Check if wanted metrics are implemented
        list_fn = [method_name for method_name in dir(MetricModule) if callable(getattr(MetricModule, method_name))]
        self.metrics = dict()
        for metric in metrics:
            if metric + '_fn' in list_fn:
                self.metrics[metric] = eval('MetricModule.' + metric + '_fn')
            else:
                raise NotImplementedError("The metric %s is not implemented in the module" % metric)

    def apply(self, labels, proba0):
        """
        This is a function to calculate the different metrics based on the list of true label and predicted label

        Args:
            labels: array of labels
            proba0: array of outputs of node 0.

        Returns:
            (dict) metrics results
        """

        if labels is not None and proba0 is not None:
            results = dict()

            for metric_key, metric_fn in self.metrics.items():
                results[metric_key] = metric_fn(labels, proba0)
        else:
            results = dict()

        return results

    @staticmethod
    def accuracy_fn(labels, proba0, threshold=0.5):
        """
        Computation of accuracy

        Args:
            labels: array of labels.
            proba0: array of outputs of node 0.
            threshold: separation of the two classes.

        Returns:
            (float) accuracy
        """
        true_positive = np.sum((labels == 1) & (proba0 < threshold))
        true_negative = np.sum((labels == 0) & (proba0 >= threshold))
        false_positive = np.sum((labels == 1) & (proba0 >= threshold))
        false_negative = np.sum((labels == 0) & (proba0 < threshold))

        return (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)

    @staticmethod
    def sensitivity_fn(labels, proba0, threshold=0.5):
        """
        Computation of sensitivity

        Args:
            labels: array of labels.
            proba0: array of outputs of node 0.
            threshold: separation of the two classes.

        Returns:
            (float) sensitivity
        """
        true_positive = np.sum((labels == 1) & (proba0 < threshold))
        false_negative = np.sum((labels == 0) & (proba0 < threshold))

        if (true_positive + false_negative) != 0:
            return true_positive / (true_positive + false_negative)
        else:
            return 0.0

    @staticmethod
    def specificity_fn(labels, proba0, threshold=0.5):
        """
        Computation of specificity

        Args:
            labels: array of labels.
            proba0: array of outputs of node 0.
            threshold: separation of the two classes.

        Returns:
            (float) specificity
        """
        true_negative = np.sum((labels == 0) & (proba0 >= threshold))
        false_positive = np.sum((labels == 1) & (proba0 >= threshold))

        if (false_positive + true_negative) != 0:
            return true_negative / (false_positive + true_negative)
        else:
            return 0.0

    @staticmethod
    def ppv_fn(labels, proba0, threshold=0.5):
        """
        Computation of positive predictive value

        Args:
            labels: array of labels.
            proba0: array of outputs of node 0.
            threshold: separation of the two classes.

        Returns:
            (float) positive predictive value
        """
        true_positive = np.sum((labels == 1) & (proba0 < threshold))
        false_positive = np.sum((labels == 1) & (proba0 >= threshold))

        if (true_positive + false_positive) != 0:
            return true_positive / (true_positive + false_positive)
        else:
            return 0.0

    @staticmethod
    def npv_fn(labels, proba0, threshold=0.5):
        """
        Computation of negative predictive value

        Args:
            labels: array of labels.
            proba0: array of outputs of node 0.
            threshold: separation of the two classes.

        Returns:
            (float) negative predictive value
        """
        true_negative = np.sum((labels == 0) & (proba0 >= threshold))
        false_negative = np.sum((labels == 0) & (proba0 < threshold))

        if (true_negative + false_negative) != 0:
            return true_negative / (true_negative + false_negative)
        else:
            return 0.0

    @staticmethod
    def balanced_accuracy_fn(labels, proba0, threshold=0.5):
        """
        Computation of the balanced accuracy

        Args:
            labels: array of labels.
            proba0: array of outputs of node 0.
            threshold: separation of the two classes.

        Returns:
            (float) balanced accuracy
        """

        return (MetricModule.sensitivity_fn(labels, proba0, threshold)
                + MetricModule.specificity_fn(labels, proba0, threshold)) / 2

    @staticmethod
    def confusion_matrix_fn(labels, proba0, threshold=0.5):
        """
        Computation of the confusion matrix

        Args:
            labels: array of labels.
            proba0: array of outputs of node 0.
            threshold: separation of the two classes.

        Returns:
            (dict) confusion matrix
        """
        true_positive = np.sum((labels == 1) & (proba0 < threshold))
        true_negative = np.sum((labels == 0) & (proba0 >= threshold))
        false_positive = np.sum((labels == 1) & (proba0 >= threshold))
        false_negative = np.sum((labels == 0) & (proba0 < threshold))

        return {'tp': true_positive, 'tn': true_negative, 'fp': false_positive, 'fn': false_negative}
