"""RobustBench ImageNet classifier used by TextBack."""

from pathlib import Path


class RobustBenchImageNetClassifier:
    """Robust ImageNet ResNet-50 loaded from RobustBench."""

    def __init__(self, model_name: str, dataset: str, threat_model: str, device: str) -> None:
        """Load the robust classifier and ImageNet preprocessing.

        Args:
            model_name: RobustBench model name, e.g. Salman2020Do_R50.
            dataset: RobustBench dataset name, here "imagenet".
            threat_model: RobustBench threat model, here "Linf".
            device: Torch device string.

        Raises:
            RuntimeError: If RobustBench imports or model loading fail.
        """
        try:
            import torch
            from PIL import Image
            from robustbench.utils import load_model
            from torchvision.models import ResNet50_Weights

            self.torch = torch
            self.Image = Image
            self.device = device

            weights = ResNet50_Weights.DEFAULT
            self.preprocess = weights.transforms()
            self.labels = weights.meta["categories"]

            self.model = load_model(
                model_name=model_name,
                dataset=dataset,
                threat_model=threat_model,
            ).to(device)
            self.model.eval()
        except Exception as error:
            raise RuntimeError(
                f"Could not load RobustBench model {model_name}. "
                "Check robustbench installation and model download."
            ) from error

    def predict(self, image_path: str | Path, target_class: str, top_k: int = 5) -> dict:
        """Classify one image and return TextBack metrics.

        Args:
            image_path: Path to the image.
            target_class: Exact ImageNet label.
            top_k: Number of top predictions to return.

        Returns:
            Dictionary with top-1, top-k, target confidence, and target rank.
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
        """Compute confidence and 1-based rank for the target class."""
        if target_class not in self.labels:
            return 0.0, None

        target_index = self.labels.index(target_class)
        target_confidence = float(probabilities[target_index].item())
        sorted_indices = self.torch.argsort(probabilities, descending=True).tolist()
        return target_confidence, sorted_indices.index(target_index) + 1


def build_classifier(config: dict) -> RobustBenchImageNetClassifier:
    """Build the final RobustBench classifier.

    Args:
        config: Loaded configuration dictionary.

    Returns:
        RobustBenchImageNetClassifier instance.

    Raises:
        ValueError: If classifier.provider is not robustbench.
    """
    classifier_config = config["classifier"]
    provider = classifier_config.get("provider")
    if provider != "robustbench":
        raise ValueError("Final project requires classifier.provider='robustbench'.")

    return RobustBenchImageNetClassifier(
        model_name=classifier_config["model_name"],
        dataset=classifier_config["dataset"],
        threat_model=classifier_config["threat_model"],
        device=config["project"].get("device", "cpu"),
    )
