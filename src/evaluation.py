"""Evaluation metrics for TextBack."""

from typing import Dict, List


def is_activation_success(classifier_output: Dict) -> bool:
    """Check whether the target class is top-1.

    Args:
        classifier_output: Output dictionary returned by the classifier.

    Returns:
        True when the target class is ranked first.
    """
    return classifier_output.get("target_rank") == 1


def activation_maximization_rate(results: List[Dict]) -> float:
    """Compute the fraction of images classified as the target class.

    Args:
        results: List of classifier output dictionaries.

    Returns:
        Success rate between 0 and 1.
    """
    if not results:
        return 0.0
    successes = sum(1 for result in results if is_activation_success(result))
    return successes / len(results)
