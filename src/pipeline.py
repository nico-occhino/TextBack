"""Main TextBack pipeline.

The optimization loop is intentionally direct:
TextGrad prompt variable -> Diffusers image -> ResNet50 prediction ->
TextGrad textual loss/backward/TGD step.
"""

import json
from pathlib import Path

from src.classifier import build_classifier
from src.config import create_output_dirs
from src.image_generator import build_image_generator
from src.logging_utils import append_csv_row, append_jsonl_record, write_json
from src.textgrad_optimizer import TextGradPromptOptimizer
from src.utils import ensure_dir, set_seed, slugify


OPTIMIZATION_COLUMNS = [
    "target_class",
    "iteration",
    "prompt",
    "textual_loss",
    "image_path",
    "top1_label",
    "top1_confidence",
    "target_confidence",
    "target_rank",
    "topk",
]

INFERENCE_COLUMNS = [
    "target_class",
    "image_index",
    "prompt",
    "image_path",
    "top1_label",
    "top1_confidence",
    "target_confidence",
    "target_rank",
    "success",
]


class TextBackPipeline:
    """Run TextGrad optimization and inference evaluation."""

    def __init__(self, config: dict) -> None:
        """Initialize TextGrad, generator, and classifier.

        Args:
            config: Loaded configuration dictionary.
        """
        self.config = config
        self.paths = config["paths"]
        self.experiment = config["experiment"]

        create_output_dirs(config)
        set_seed(int(config["project"].get("seed", 42)))

        self.textgrad_optimizer = TextGradPromptOptimizer(config)
        self.image_generator = build_image_generator(config)
        self.classifier = build_classifier(config)

    def run_optimization(self) -> dict[str, str]:
        """Optimize prompts for all configured target classes.

        Returns:
            Mapping from target class to final prompt.
        """
        self._reset_optimization_logs()
        final_prompts = {}

        for target_class in self.experiment["target_classes"]:
            print(f"Optimizing prompt for: {target_class}")
            final_prompts[target_class] = self._optimize_one_class(target_class)

        self._save_final_prompts(final_prompts)
        return final_prompts

    def _optimize_one_class(self, target_class: str) -> str:
        """Run TextGrad prompt optimization for one class.

        Args:
            target_class: Exact ImageNet target label.

        Returns:
            Final optimized prompt.
        """
        initial_prompt = self._make_initial_prompt(target_class)
        prompt_variable = self.textgrad_optimizer.make_prompt_variable(initial_prompt, target_class)
        updated_prompt = self._textgrad_value(prompt_variable)

        for iteration in range(int(self.experiment["n_optimization_steps"])):
            prompt = self._textgrad_value(prompt_variable)
            image_path = self._optimization_image_path(target_class, iteration)

            self.image_generator.generate(prompt, image_path)
            classifier_result = self.classifier.predict(
                image_path=image_path,
                target_class=target_class,
                top_k=int(self.experiment.get("top_k", 5)),
            )

            updated_prompt, textual_loss = self.textgrad_optimizer.step(
                prompt_variable,
                target_class,
                classifier_result,
            )

            self._log_optimization_step(
                target_class,
                iteration,
                prompt,
                textual_loss,
                image_path,
                classifier_result,
            )

        return updated_prompt

    def _make_initial_prompt(self, target_class: str) -> str:
        """Create a shortcut-search prompt that does not name the target class.

        Args:
            target_class: Hidden classifier target, not included in the prompt.

        Returns:
            Initial image-generation prompt.
        """
        return (
            "A realistic photographic scene containing indirect visual cues such as "
            "textures, colors, lighting, backgrounds, and contextual patterns that "
            "may be statistically associated with the hidden target category, without "
            "showing or naming the target object."
        )

    def run_inference(self) -> dict[str, float]:
        """Generate images from final prompts and evaluate activation rate.

        Returns:
            Mapping from target class to activation maximization rate.
        """
        self._reset_inference_logs()
        final_prompts = self._load_final_prompts()
        activation_rates = {}

        for target_class, prompt in final_prompts.items():
            print(f"Running inference for: {target_class}")
            successes = 0
            n_images = int(self.experiment["n_inference_images"])

            for image_index in range(n_images):
                image_path = self._inference_image_path(target_class, image_index)
                self.image_generator.generate(prompt, image_path)

                classifier_result = self.classifier.predict(
                    image_path=image_path,
                    target_class=target_class,
                    top_k=int(self.experiment.get("top_k", 5)),
                )

                success = classifier_result["target_rank"] == 1
                successes += int(success)
                self._log_inference_result(target_class, image_index, prompt, image_path, classifier_result, success)

            activation_rates[target_class] = successes / n_images if n_images > 0 else 0.0

        write_json(Path(self.paths["results_dir"]) / "activation_rates.json", activation_rates)
        return activation_rates

    def _optimization_image_path(self, target_class: str, iteration: int) -> Path:
        """Build the optimization image path for one iteration."""
        image_dir = Path(self.paths["generated_images_dir"]) / slugify(target_class) / "optimization"
        ensure_dir(image_dir)
        return image_dir / f"step_{iteration:03d}.png"

    def _inference_image_path(self, target_class: str, image_index: int) -> Path:
        """Build the inference image path for one generated sample."""
        image_dir = Path(self.paths["generated_images_dir"]) / slugify(target_class) / "inference"
        ensure_dir(image_dir)
        return image_dir / f"sample_{image_index:03d}.png"

    def _log_optimization_step(
        self,
        target_class: str,
        iteration: int,
        prompt: str,
        textual_loss: str,
        image_path: Path,
        classifier_result: dict,
    ) -> None:
        """Save one optimization step to CSV and JSONL."""
        row = {
            "target_class": target_class,
            "iteration": iteration,
            "prompt": prompt,
            "textual_loss": textual_loss,
            "image_path": str(image_path),
            "top1_label": classifier_result["top1_label"],
            "top1_confidence": classifier_result["top1_confidence"],
            "target_confidence": classifier_result["target_confidence"],
            "target_rank": classifier_result["target_rank"],
            "topk": classifier_result["topk"],
        }

        results_dir = Path(self.paths["results_dir"])
        append_csv_row(results_dir / "optimization_logs.csv", row, OPTIMIZATION_COLUMNS)
        append_jsonl_record(results_dir / "optimization_logs.jsonl", row)

    def _log_inference_result(
        self,
        target_class: str,
        image_index: int,
        prompt: str,
        image_path: Path,
        classifier_result: dict,
        success: bool,
    ) -> None:
        """Save one inference image result to CSV."""
        row = {
            "target_class": target_class,
            "image_index": image_index,
            "prompt": prompt,
            "image_path": str(image_path),
            "top1_label": classifier_result["top1_label"],
            "top1_confidence": classifier_result["top1_confidence"],
            "target_confidence": classifier_result["target_confidence"],
            "target_rank": classifier_result["target_rank"],
            "success": success,
        }

        append_csv_row(Path(self.paths["results_dir"]) / "inference_results.csv", row, INFERENCE_COLUMNS)

    def _save_final_prompts(self, final_prompts: dict[str, str]) -> None:
        """Save final prompts after optimization."""
        write_json(Path(self.paths["results_dir"]) / "final_prompts.json", final_prompts)

    def _reset_optimization_logs(self) -> None:
        """Remove old optimization outputs before a new run."""
        results_dir = Path(self.paths["results_dir"])
        for file_name in ["final_prompts.json", "optimization_logs.csv", "optimization_logs.jsonl"]:
            path = results_dir / file_name
            if path.exists():
                path.unlink()

    def _reset_inference_logs(self) -> None:
        """Remove old inference outputs before a new run."""
        results_dir = Path(self.paths["results_dir"])
        for file_name in ["inference_results.csv", "activation_rates.json", "inference_metrics.json"]:
            path = results_dir / file_name
            if path.exists():
                path.unlink()

    def _load_final_prompts(self) -> dict[str, str]:
        """Load prompts saved by run_optimization()."""
        path = Path(self.paths["results_dir"]) / "final_prompts.json"
        if not path.exists():
            raise FileNotFoundError("Run optimization first: results/final_prompts.json is missing.")

        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _textgrad_value(self, variable) -> str:
        """Read a TextGrad variable value across TextGrad versions."""
        if hasattr(variable, "value"):
            return str(variable.value)
        if hasattr(variable, "get_value"):
            return str(variable.get_value())
        return str(variable)
