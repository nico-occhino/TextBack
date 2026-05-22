"""ImageNet classifier wrappers for TextBack."""

from pathlib import Path


class DummyImageNetClassifier:
    """Small deterministic classifier used in dry-run mode."""

    def __init__(self) -> None:
        """Create a few distractor labels for readable fake predictions."""
        self.distractor_labels = ["minibus", "moving van", "tabby cat", "sports car", "traffic light"]

    def predict(self, image_path: str | Path, target_class: str, top_k: int = 5) -> dict:
        """Return fake ImageNet predictions.

        Args:
            image_path: Path to the generated image.
            target_class: Desired ImageNet class.
            top_k: Number of top predictions to return.

        Returns:
            Dictionary with top-1, top-k, target confidence, and target rank.
        """
        image_index = self._index_from_path(image_path)
        target_confidence = min(0.18 + 0.22 * image_index, 0.93)
        distractor_confidence = max(0.82 - 0.18 * image_index, 0.04)

        topk = [
            {"label": self.distractor_labels[image_index % len(self.distractor_labels)], "confidence": distractor_confidence, "index": 0},
            {"label": target_class, "confidence": target_confidence, "index": 1},
            {"label": "background", "confidence": 0.08, "index": 2},
            {"label": "texture", "confidence": 0.05, "index": 3},
            {"label": "object", "confidence": 0.03, "index": 4},
        ]
        topk = sorted(topk, key=lambda item: item["confidence"], reverse=True)[:top_k]

        target_rank = self._find_rank(topk, target_class)
        return {
            "top1_label": topk[0]["label"],
            "top1_confidence": topk[0]["confidence"],
            "topk": topk,
            "target_confidence": target_confidence,
            "target_rank": target_rank,
        }

    def _index_from_path(self, image_path: str | Path) -> int:
        """Extract a numeric index from names like step_002.png.

        Args:
            image_path: Image path.

        Returns:
            Parsed index or 0 when no number is present.
        """
        digits = "".join(character for character in Path(image_path).stem if character.isdigit())
        return int(digits) if digits else 0

    def _find_rank(self, topk: list[dict], target_class: str) -> int | None:
        """Find the 1-based rank of the target in a top-k list.

        Args:
            topk: Ordered prediction dictionaries.
            target_class: Desired class label.

        Returns:
            Rank if present, otherwise None.
        """
        for rank, prediction in enumerate(topk, start=1):
            if prediction["label"] == target_class:
                return rank
        return None


class TorchvisionImageNetClassifier:
    """Torchvision ResNet50 classifier with ImageNet preprocessing."""

    def __init__(self, device: str = "cpu") -> None:
        """Load pretrained ResNet50 and its matching preprocessing transform.

        Args:
            device: Torch device string, usually "cpu" or "cuda".
        """
        import torch
        from PIL import Image
        from torchvision.models import ResNet50_Weights, resnet50

        self.torch = torch
        self.Image = Image
        self.device = device

        # Torchvision weights include the correct resize, crop, normalize steps.
        self.weights = ResNet50_Weights.DEFAULT
        self.preprocess = self.weights.transforms()
        self.labels = self.weights.meta["categories"]

        self.model = resnet50(weights=self.weights).to(self.device)
        self.model.eval()

    def predict(self, image_path: str | Path, target_class: str, top_k: int = 5) -> dict:
        """Classify an image and compute target-class metrics.

        Args:
            image_path: Path to the input image.
            target_class: Desired ImageNet label.
            top_k: Number of top predictions to return.

        Returns:
            Dictionary with top1_label, top1_confidence, topk, target_confidence,
            and target_rank.  If target_class is not an exact ImageNet label,
            target_confidence is 0 and target_rank is None.
        """
        image = self.Image.open(image_path).convert("RGB")
        input_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with self.torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = self.torch.softmax(logits, dim=1)[0]

        top_values, top_indices = probabilities.topk(top_k)
        topk = []
        for confidence, index in zip(top_values.tolist(), top_indices.tolist()):
            topk.append(
                {
                    "label": self.labels[index],
                    "confidence": float(confidence),
                    "index": int(index),
                }
            )

        target_confidence, target_rank = self._target_metrics(probabilities, target_class)
        return {
            "top1_label": topk[0]["label"],
            "top1_confidence": topk[0]["confidence"],
            "topk": topk,
            "target_confidence": target_confidence,
            "target_rank": target_rank,
        }

    def _target_metrics(self, probabilities, target_class: str) -> tuple[float, int | None]:
        """Compute confidence and rank for an exact ImageNet label.

        Args:
            probabilities: Tensor of ImageNet probabilities.
            target_class: Desired ImageNet label.

        Returns:
            Target confidence and rank, or (0.0, None) if the label is unknown.
        """
        if target_class not in self.labels:
            return 0.0, None

        target_index = self.labels.index(target_class)
        target_confidence = float(probabilities[target_index].item())
        sorted_indices = probabilities.argsort(descending=True).tolist()
        target_rank = sorted_indices.index(target_index) + 1
        return target_confidence, target_rank


def build_classifier(config: dict):
    """Create the classifier selected by the config.

    Args:
        config: Loaded project configuration.

    Returns:
        A classifier object with a predict() method.
    """
    dry_run = bool(config.get("project", {}).get("dry_run", True))
    provider = config.get("classifier", {}).get("provider", "mock")

    if dry_run or provider in {"mock", "dummy"}:
        return DummyImageNetClassifier()

    if provider == "torchvision":
        device = config.get("project", {}).get("device", "cpu")
        return TorchvisionImageNetClassifier(device=device)

    print("Unknown classifier provider. Falling back to dummy classifier.")
    return DummyImageNetClassifier()
