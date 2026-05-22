"""Classifier loading, preprocessing, and prediction helpers.

The dry-run implementation uses a mock classifier. A lightweight torchvision
hook is included so the project can be extended later without changing the
pipeline code.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class Prediction:
    """One classifier prediction.

    Attributes:
        label: Human readable class label.
        confidence: Confidence score between 0 and 1.
    """

    label: str
    confidence: float


class BaseClassifier:
    """Minimal classifier interface used by optimization and inference."""

    def classify(self, image_path: str | Path, target_class: str, top_k: int = 5) -> Dict:
        """Classify one image and report target-specific metrics.

        Args:
            image_path: Path to the generated image.
            target_class: Target ImageNet class.
            top_k: Number of predictions to return.

        Returns:
            Dictionary with top predictions, target confidence, and target rank.
        """
        raise NotImplementedError


class MockClassifier(BaseClassifier):
    """Deterministic classifier used to test the pipeline without torch."""

    def __init__(self) -> None:
        """Create a small list of ImageNet-like labels."""
        self.background_labels = [
            "minibus",
            "moving van",
            "sports car",
            "tabby cat",
            "tennis ball",
            "lakeside",
            "traffic light",
        ]

    def classify(self, image_path: str | Path, target_class: str, top_k: int = 5) -> Dict:
        """Return fake predictions that improve over optimization steps.

        Args:
            image_path: Path to the generated image.
            target_class: Target ImageNet class.
            top_k: Number of predictions to return.

        Returns:
            Dictionary with top-k predictions and target metrics.
        """
        step = self._step_from_path(image_path)
        target_confidence = min(0.15 + 0.22 * step, 0.92)
        distractor_confidence = max(0.80 - 0.18 * step, 0.05)

        predictions = [
            Prediction(self.background_labels[step % len(self.background_labels)], distractor_confidence),
            Prediction(target_class, target_confidence),
            Prediction("background", 0.10),
            Prediction("object", 0.08),
            Prediction("texture", 0.05),
        ]
        predictions = sorted(predictions, key=lambda item: item.confidence, reverse=True)
        predictions = predictions[:top_k]

        target_rank = self._find_target_rank(predictions, target_class)
        return {
            "top_predictions": [asdict(prediction) for prediction in predictions],
            "target_confidence": float(target_confidence),
            "target_rank": target_rank,
        }

    def _step_from_path(self, image_path: str | Path) -> int:
        """Extract an optimization or inference index from a file name.

        Args:
            image_path: Image path such as step_002.png.

        Returns:
            Parsed integer index, or 0 if parsing fails.
        """
        stem = Path(image_path).stem
        digits = "".join(character for character in stem if character.isdigit())
        return int(digits) if digits else 0

    def _find_target_rank(self, predictions: List[Prediction], target_class: str) -> int | None:
        """Find the 1-based target rank in top-k predictions.

        Args:
            predictions: Ordered prediction objects.
            target_class: Target ImageNet class.

        Returns:
            Rank if the target appears, otherwise None.
        """
        for index, prediction in enumerate(predictions, start=1):
            if prediction.label == target_class:
                return index
        return None


class TorchvisionClassifier(BaseClassifier):
    """Small wrapper around torchvision ResNet50.

    This class is intentionally lazy: heavy imports happen only when the user
    requests the real classifier. Dry-run mode does not need torch installed.
    """

    def __init__(self, config: Dict) -> None:
        """Load a pretrained torchvision classifier.

        Args:
            config: Loaded configuration dictionary.
        """
        import torch
        from PIL import Image
        from torchvision.models import ResNet50_Weights, resnet50

        self.torch = torch
        self.Image = Image
        self.device = config.get("project", {}).get("device", "cpu")
        weights = ResNet50_Weights.DEFAULT
        self.categories = weights.meta["categories"]
        self.preprocess = weights.transforms()
        self.model = resnet50(weights=weights).to(self.device)
        self.model.eval()

    def classify(self, image_path: str | Path, target_class: str, top_k: int = 5) -> Dict:
        """Classify one image with ResNet50.

        Args:
            image_path: Path to the generated image.
            target_class: Target ImageNet class.
            top_k: Number of predictions to return.

        Returns:
            Dictionary with top-k predictions and target metrics.
        """
        image = self.Image.open(image_path).convert("RGB")
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with self.torch.no_grad():
            logits = self.model(tensor)
            probabilities = self.torch.softmax(logits, dim=1)[0]

        top_values, top_indices = probabilities.topk(top_k)
        top_predictions = []
        for value, index in zip(top_values.tolist(), top_indices.tolist()):
            top_predictions.append({"label": self.categories[index], "confidence": float(value)})

        target_confidence, target_rank = self._target_metrics(probabilities, target_class)
        return {
            "top_predictions": top_predictions,
            "target_confidence": target_confidence,
            "target_rank": target_rank,
        }

    def _target_metrics(self, probabilities, target_class: str) -> tuple[float, int | None]:
        """Compute confidence and rank for the target class.

        Args:
            probabilities: Torch tensor with ImageNet probabilities.
            target_class: Human readable target class.

        Returns:
            Target confidence and 1-based rank, if the class is known.
        """
        if target_class not in self.categories:
            return 0.0, None

        target_index = self.categories.index(target_class)
        target_confidence = float(probabilities[target_index].item())
        sorted_indices = probabilities.argsort(descending=True).tolist()
        return target_confidence, sorted_indices.index(target_index) + 1


def build_classifier(config: Dict) -> BaseClassifier:
    """Create the classifier requested by the config.

    Args:
        config: Loaded configuration dictionary.

    Returns:
        A classifier object.
    """
    provider = config.get("classifier", {}).get("provider", "mock")
    dry_run = config.get("project", {}).get("dry_run", True)

    if dry_run or provider == "mock":
        return MockClassifier()
    if provider == "torchvision":
        return TorchvisionClassifier(config)

    print("Unknown classifier provider; falling back to mock classifier.")
    return MockClassifier()
