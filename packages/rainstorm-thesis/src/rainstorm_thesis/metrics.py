"""Small dependency-free metrics for the bundled NOR benchmark."""
from __future__ import annotations

from collections import Counter
from math import log


class BinaryMetric:
    compatible_task = 'classification'

    def evaluate(self, *, dataset, output, model):
        targets, predictions = _paired_binary(dataset, output)
        true_positive = sum(target == 1 and prediction == 1
                            for target, prediction in zip(targets, predictions, strict=True))
        false_positive = sum(target == 0 and prediction == 1
                             for target, prediction in zip(targets, predictions, strict=True))
        false_negative = sum(target == 1 and prediction == 0
                             for target, prediction in zip(targets, predictions, strict=True))
        true_negative = sum(target == 0 and prediction == 0
                            for target, prediction in zip(targets, predictions, strict=True))
        return self.score(true_positive, false_positive, false_negative, true_negative)


class Precision(BinaryMetric):
    def score(self, tp, fp, _fn, _tn):
        return tp / (tp + fp) if tp + fp else 0.0


class Recall(BinaryMetric):
    def score(self, tp, _fp, fn, _tn):
        return tp / (tp + fn) if tp + fn else 0.0


class F1(BinaryMetric):
    def score(self, tp, fp, fn, _tn):
        denominator = 2 * tp + fp + fn
        return 2 * tp / denominator if denominator else 0.0


class BalancedAccuracy(BinaryMetric):
    def score(self, tp, fp, fn, tn):
        recalls = []
        if tp + fn:
            recalls.append(tp / (tp + fn))
        if tn + fp:
            recalls.append(tn / (tn + fp))
        return sum(recalls) / len(recalls) if recalls else 0.0


class ClusteringMetric:
    compatible_task = 'clustering'

    def evaluate(self, *, dataset, output, model):
        targets, predictions = _paired_values(dataset, output)
        return self.score(targets, predictions)


class AdjustedRandIndex(ClusteringMetric):
    def score(self, targets, predictions):
        if len(targets) < 2:
            return 1.0
        table, target_counts, prediction_counts = _contingency(targets, predictions)
        choose_two = lambda value: value * (value - 1) / 2
        cell_pairs = sum(choose_two(count) for row in table.values()
                         for count in row.values())
        target_pairs = sum(choose_two(count) for count in target_counts.values())
        prediction_pairs = sum(choose_two(count) for count in prediction_counts.values())
        total_pairs = choose_two(len(targets))
        expected = target_pairs * prediction_pairs / total_pairs
        maximum = (target_pairs + prediction_pairs) / 2
        if maximum == expected:
            return 1.0
        return (cell_pairs - expected) / (maximum - expected)


class NormalizedMutualInformation(ClusteringMetric):
    def score(self, targets, predictions):
        table, target_counts, prediction_counts = _contingency(targets, predictions)
        target_entropy = _entropy(target_counts.values(), len(targets))
        prediction_entropy = _entropy(prediction_counts.values(), len(predictions))
        if target_entropy == 0 and prediction_entropy == 0:
            return 1.0
        if target_entropy == 0 or prediction_entropy == 0:
            return 0.0
        mutual_information = _mutual_information(
            table, target_counts, prediction_counts, len(targets))
        return 2 * mutual_information / (target_entropy + prediction_entropy)


class Homogeneity(ClusteringMetric):
    def score(self, targets, predictions):
        table, target_counts, prediction_counts = _contingency(targets, predictions)
        entropy = _entropy(target_counts.values(), len(targets))
        if entropy == 0:
            return 1.0
        return _mutual_information(table, target_counts, prediction_counts,
                                   len(targets)) / entropy


class Completeness(ClusteringMetric):
    def score(self, targets, predictions):
        table, target_counts, prediction_counts = _contingency(targets, predictions)
        entropy = _entropy(prediction_counts.values(), len(predictions))
        if entropy == 0:
            return 1.0
        return _mutual_information(table, target_counts, prediction_counts,
                                   len(targets)) / entropy


def register_metrics(registry):
    """Register benchmark metrics while declaring which output they score."""
    for name, metric in (
        ('precision', Precision()), ('recall', Recall()), ('f1', F1()),
        ('balanced_accuracy', BalancedAccuracy()),
        ('ari', AdjustedRandIndex()), ('nmi', NormalizedMutualInformation()),
        ('homogeneity', Homogeneity()), ('completeness', Completeness()),
    ):
        registry.register(name, metric, direction='maximize')


def _paired_values(dataset, output):
    if dataset.targets is None:
        raise ValueError('This metric requires reference labels.')
    targets, predictions = list(dataset.targets), list(output.predictions)
    if not targets or len(targets) != len(predictions):
        raise ValueError('Predictions and reference labels must be nonempty and aligned.')
    return targets, predictions


def _paired_binary(dataset, output):
    targets, predictions = _paired_values(dataset, output)
    if any(value not in (0, 1) or isinstance(value, bool)
           for value in targets + predictions):
        raise ValueError('Binary classification metrics require labels coded as 0 or 1.')
    return targets, predictions


def _contingency(targets, predictions):
    if len(targets) != len(predictions) or not targets:
        raise ValueError('Cluster predictions and reference labels must be aligned.')
    target_counts = Counter(targets)
    prediction_counts = Counter(predictions)
    table = {}
    for target, prediction in zip(targets, predictions, strict=True):
        row = table.setdefault(target, Counter())
        row[prediction] += 1
    return table, target_counts, prediction_counts


def _entropy(counts, total):
    return -sum((count / total) * log(count / total)
                for count in counts if count)


def _mutual_information(table, target_counts, prediction_counts, total):
    result = 0.0
    for target, row in table.items():
        for prediction, count in row.items():
            result += (count / total) * log(
                count * total / (target_counts[target] * prediction_counts[prediction]))
    return result
