"""Torchvision ImageNet classifier used by TextBack."""

from pathlib import Path


class TorchvisionImageNetClassifier:
    """ResNet50 classifier with official ImageNet preprocessing."""

    def __init__(self, device: str = "cpu") -> None:
        """Load pretrained ResNet50 and its matching transforms.

        Args:
            device: Torch device string, usually "cpu" or "cuda".
        """
        import torch
        from PIL import Image
        from torchvision.models import ResNet50_Weights, resnet50

        self.torch = torch
        self.Image = Image
        self.device = device

        self.weights = ResNet50_Weights.DEFAULT
        self.preprocess = self.weights.transforms()
        self.labels = self.weights.meta["categories"]

        self.model = resnet50(weights=self.weights).to(self.device)
        self.model.eval()

    def predict(self, image_path: str | Path, target_class: str, top_k: int = 5) -> dict:
        """Classify one image and compute target-class metrics.

        Args:
            image_path: Path to the image.
            target_class: Exact ImageNet label we want to activate.
            top_k: Number of top predictions to return.

        Returns:
            Dictionary with top-1, top-k, target confidence, and target rank.
            If target_class is not an exact ImageNet label, target rank is None.
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
            target_class: Exact target label.

        Returns:
            Target confidence and 1-based rank, or (0.0, None) if unknown.
        """
        if target_class not in self.labels:
            return 0.0, None

        target_index = self.labels.index(target_class)
        target_confidence = float(probabilities[target_index].item())
        sorted_indices = probabilities.argsort(descending=True).tolist()
        return target_confidence, sorted_indices.index(target_index) + 1


def build_classifier(config: dict) -> TorchvisionImageNetClassifier:
    """Create the configured classifier.

    Args:
        config: Loaded project configuration.

    Returns:
        TorchvisionImageNetClassifier instance.

    Raises:
        ValueError: If classifier.provider is not "torchvision".
    """
    provider = config.get("classifier", {}).get("provider")
    if provider != "torchvision":
        raise ValueError(f"Unsupported classifier provider: {provider}. Use 'torchvision'.")

    device = config.get("project", {}).get("device", "cpu")
    return TorchvisionImageNetClassifier(device=device)
