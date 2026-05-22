"""Main orchestration code for optimization and inference."""

from pathlib import Path
from typing import Dict

from src.classifier import build_classifier
from src.config import prepare_output_dirs
from src.evaluation import activation_maximization_rate
from src.image_generator import build_image_generator
from src.llm_client import build_llm_client
from src.logging_utils import append_csv, append_jsonl, write_json
from src.textual_backward import generate_initial_prompt, refine_prompt
from src.utils import ensure_dir, set_seed, slugify


OPTIMIZATION_FIELDS = [
    "class_name",
    "step",
    "prompt",
    "image_path",
    "target_confidence",
    "target_rank",
    "top_predictions",
]


def run_optimization(config: Dict) -> Dict[str, str]:
    """Run the textual backward optimization loop.

    Args:
        config: Loaded configuration dictionary.

    Returns:
        Mapping from class name to final optimized prompt.
    """
    prepare_output_dirs(config)
    set_seed(int(config["project"].get("seed", 42)))

    llm_client = build_llm_client(config)
    image_generator = build_image_generator(config)
    classifier = build_classifier(config)

    final_prompts = {}
    for class_name in config["experiment"]["target_classes"]:
        final_prompts[class_name] = _optimize_one_class(
            config,
            class_name,
            llm_client,
            image_generator,
            classifier,
        )

    final_path = Path(config["paths"]["results_dir"]) / "final_prompts.json"
    write_json(final_path, final_prompts)
    return final_prompts


def _optimize_one_class(config: Dict, class_name: str, llm_client, image_generator, classifier) -> str:
    """Optimize a prompt for one ImageNet class.

    Args:
        config: Loaded configuration dictionary.
        class_name: Target ImageNet class.
        llm_client: Prompt generation client.
        image_generator: Text-to-image generator.
        classifier: Image classifier.

    Returns:
        Final prompt after all optimization steps.
    """
    prompt = generate_initial_prompt(llm_client, class_name)
    n_steps = int(config["experiment"]["n_optimization_steps"])
    top_k = int(config["experiment"].get("top_k", 5))

    for step in range(n_steps):
        image_path = _optimization_image_path(config, class_name, step)
        image_generator.generate(prompt, image_path)
        classifier_output = classifier.classify(image_path, class_name, top_k=top_k)

        _log_optimization_step(config, class_name, step, prompt, image_path, classifier_output)
        prompt = refine_prompt(llm_client, class_name, prompt, classifier_output, step + 1)

    return prompt


def _optimization_image_path(config: Dict, class_name: str, step: int) -> Path:
    """Build the output path for one optimization image.

    Args:
        config: Loaded configuration dictionary.
        class_name: Target ImageNet class.
        step: Optimization step index.

    Returns:
        Path where the image should be saved.
    """
    class_dir = Path(config["paths"]["generated_images_dir"]) / slugify(class_name) / "optimization"
    ensure_dir(class_dir)
    return class_dir / f"step_{step:03d}.png"


def _log_optimization_step(
    config: Dict,
    class_name: str,
    step: int,
    prompt: str,
    image_path: Path,
    classifier_output: Dict,
) -> None:
    """Write one optimization step to CSV and JSONL logs.

    Args:
        config: Loaded configuration dictionary.
        class_name: Target ImageNet class.
        step: Optimization step index.
        prompt: Prompt used for the generated image.
        image_path: Saved image path.
        classifier_output: Output dictionary returned by the classifier.
    """
    results_dir = Path(config["paths"]["results_dir"])
    row = {
        "class_name": class_name,
        "step": step,
        "prompt": prompt,
        "image_path": str(image_path),
        "target_confidence": classifier_output["target_confidence"],
        "target_rank": classifier_output["target_rank"],
        "top_predictions": classifier_output["top_predictions"],
    }

    append_csv(results_dir / "optimization_logs.csv", row, OPTIMIZATION_FIELDS)
    append_jsonl(results_dir / "optimization_logs.jsonl", row)


def run_inference(config: Dict) -> Dict[str, Dict]:
    """Generate images from final prompts and compute success rates.

    Args:
        config: Loaded configuration dictionary.

    Returns:
        Per-class inference metrics.
    """
    prepare_output_dirs(config)
    image_generator = build_image_generator(config)
    classifier = build_classifier(config)

    final_prompts_path = Path(config["paths"]["results_dir"]) / "final_prompts.json"
    final_prompts = _load_final_prompts(final_prompts_path)

    metrics = {}
    for class_name, prompt in final_prompts.items():
        metrics[class_name] = _run_inference_for_class(config, class_name, prompt, image_generator, classifier)

    metrics_path = Path(config["paths"]["results_dir"]) / "inference_metrics.json"
    write_json(metrics_path, metrics)
    return metrics


def _load_final_prompts(path: Path) -> Dict[str, str]:
    """Load final prompts saved by optimization.

    Args:
        path: Path to results/final_prompts.json.

    Returns:
        Mapping from class name to final prompt.
    """
    import json

    if not path.exists():
        raise FileNotFoundError(
            f"Final prompts were not found at {path}. Run scripts/run_optimization.py first."
        )

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _run_inference_for_class(config: Dict, class_name: str, prompt: str, image_generator, classifier) -> Dict:
    """Run inference images for one final prompt.

    Args:
        config: Loaded configuration dictionary.
        class_name: Target ImageNet class.
        prompt: Final optimized prompt.
        image_generator: Text-to-image generator.
        classifier: Image classifier.

    Returns:
        Dictionary with activation maximization rate and image-level results.
    """
    n_images = int(config["experiment"]["n_inference_images"])
    top_k = int(config["experiment"].get("top_k", 5))
    image_results = []

    for index in range(n_images):
        image_path = _inference_image_path(config, class_name, index)
        image_generator.generate(prompt, image_path)
        classifier_output = classifier.classify(image_path, class_name, top_k=top_k)
        classifier_output["image_path"] = str(image_path)
        image_results.append(classifier_output)

    return {
        "prompt": prompt,
        "n_images": n_images,
        "activation_maximization_rate": activation_maximization_rate(image_results),
        "images": image_results,
    }


def _inference_image_path(config: Dict, class_name: str, index: int) -> Path:
    """Build the output path for one inference image.

    Args:
        config: Loaded configuration dictionary.
        class_name: Target ImageNet class.
        index: Inference image index.

    Returns:
        Path where the inference image should be saved.
    """
    class_dir = Path(config["paths"]["generated_images_dir"]) / slugify(class_name) / "inference"
    ensure_dir(class_dir)
    return class_dir / f"sample_{index:03d}.png"
