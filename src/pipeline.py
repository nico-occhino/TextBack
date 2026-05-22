"""Main TextBack pipeline.

The pipeline keeps the experiment loop explicit: prompt, image, classifier,
textual feedback, prompt update.  This is easier to discuss and modify during
an oral exam than a heavily abstracted framework.
"""

import json
from pathlib import Path

from src.classifier import build_classifier
from src.config import create_output_dirs
from src.image_generator import build_image_generator
from src.llm_client import DummyLLMClient
from src.logging_utils import append_csv_row, append_jsonl_record, write_json
from src.textual_backward import TextualBackwardOptimizer
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
    """Orchestrates prompt optimization and inference evaluation."""

    def __init__(self, config: dict) -> None:
        """Initialize all components from the config.

        Args:
            config: Loaded configuration dictionary.
        """
        self.config = config
        self.paths = config["paths"]
        self.experiment = config["experiment"]

        create_output_dirs(config)
        set_seed(int(config.get("project", {}).get("seed", 42)))

        self.llm_client = DummyLLMClient()
        self.image_generator = build_image_generator(config)
        self.classifier = build_classifier(config)
        self.textual_optimizer = TextualBackwardOptimizer(
            llm_client=self.llm_client,
            prompts_dir=self.paths["prompts_dir"],
        )

    def run_optimization(self) -> dict[str, str]:
        """Run prompt optimization for every configured target class.

        Returns:
            Dictionary mapping each target class to its final prompt.
        """
        self._reset_optimization_logs()
        final_prompts = {}

        for target_class in self.experiment["target_classes"]:
            print(f"Optimizing prompt for: {target_class}")
            prompt = self.textual_optimizer.initial_prompt(target_class)

            for iteration in range(int(self.experiment["n_optimization_steps"])):
                image_path = self._optimization_image_path(target_class, iteration)

                # Forward pass: prompt -> image -> classifier prediction.
                self.image_generator.generate(prompt, image_path)
                classifier_result = self.classifier.predict(
                    image_path=image_path,
                    target_class=target_class,
                    top_k=int(self.experiment.get("top_k", 5)),
                )

                # Backward/update signal: classifier result -> textual loss.
                textual_loss = self.textual_optimizer.textual_loss(
                    target_class,
                    prompt,
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

                prompt = self.textual_optimizer.refine(target_class, prompt, classifier_result)

            final_prompts[target_class] = prompt

        self._save_final_prompts(final_prompts)
        return final_prompts

    def run_inference(self) -> dict[str, float]:
        """Evaluate final prompts with multiple generated images.

        Returns:
            Dictionary mapping each target class to activation maximization rate.
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
        """Build an optimization image path for a class and iteration.

        Args:
            target_class: Desired ImageNet class.
            iteration: Optimization step index.

        Returns:
            Path where the generated image should be saved.
        """
        image_dir = Path(self.paths["generated_images_dir"]) / slugify(target_class) / "optimization"
        ensure_dir(image_dir)
        return image_dir / f"step_{iteration:03d}.png"

    def _inference_image_path(self, target_class: str, image_index: int) -> Path:
        """Build an inference image path for a class and sample index.

        Args:
            target_class: Desired ImageNet class.
            image_index: Inference sample index.

        Returns:
            Path where the generated image should be saved.
        """
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
        """Save one optimization step to CSV and JSONL logs.

        Args:
            target_class: Desired ImageNet class.
            iteration: Optimization step index.
            prompt: Prompt used for the image.
            textual_loss: Natural language feedback.
            image_path: Saved image path.
            classifier_result: Output from classifier.predict().
        """
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
        """Save one inference image result to CSV.

        Args:
            target_class: Desired ImageNet class.
            image_index: Inference sample index.
            prompt: Final prompt used to generate the image.
            image_path: Saved image path.
            classifier_result: Output from classifier.predict().
            success: True when the target class is top-1.
        """
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
        """Save final prompts after optimization.

        Args:
            final_prompts: Mapping from target class to final prompt.
        """
        write_json(Path(self.paths["results_dir"]) / "final_prompts.json", final_prompts)

    def _reset_optimization_logs(self) -> None:
        """Remove old optimization logs before a new run."""
        results_dir = Path(self.paths["results_dir"])
        for file_name in ["optimization_logs.csv", "optimization_logs.jsonl"]:
            path = results_dir / file_name
            if path.exists():
                path.unlink()

    def _reset_inference_logs(self) -> None:
        """Remove old inference logs before a new run."""
        path = Path(self.paths["results_dir"]) / "inference_results.csv"
        if path.exists():
            path.unlink()

    def _load_final_prompts(self) -> dict[str, str]:
        """Load final prompts produced by run_optimization().

        Returns:
            Mapping from target class to final prompt.
        """
        path = Path(self.paths["results_dir"]) / "final_prompts.json"
        if not path.exists():
            raise FileNotFoundError("Run optimization first: results/final_prompts.json is missing.")

        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
