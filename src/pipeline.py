"""Main TextBack pipeline.

The optimization loop is intentionally direct:
TextGrad prompt variable -> Diffusers image -> RobustResNet50 prediction ->
TextGrad textual loss/backward/TGD step.
"""

import json
from pathlib import Path
import statistics

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
    "candidate_prompt",
    "accepted_prompt",
    "forbidden_terms",
    "was_rejected",
    "seed",
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
    "seed",
    "image_path",
    "top1_label",
    "top1_confidence",
    "target_confidence",
    "target_rank",
    "top1_correct",
    "top5_correct",
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

        # Load RobustBench before TextGrad/LiteLLM to avoid a gdown/importlib
        # metadata issue observed on Windows.
        self.classifier = build_classifier(config)
        self.image_generator = build_image_generator(config)
        self.textgrad_optimizer = TextGradPromptOptimizer(config)
        self.best_prompt_metadata = {}

    def run_optimization(self) -> dict[str, str]:
        """Optimize prompts for all configured target classes.

        Returns:
            Mapping from target class to final prompt.
        """
        self._reset_optimization_logs()
        final_prompts = {}
        self.best_prompt_metadata = {}

        for target_class in self.experiment["target_classes"]:
            print(f"Optimizing prompt for: {target_class}")
            try:
                final_prompts[target_class] = self._optimize_one_class(target_class)
                self._save_final_prompts(final_prompts)
            except RuntimeError:
                self._save_final_prompts(final_prompts)
                print("Stopping optimization early. Partial prompts were saved.")
                raise

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
        optimizer = self.textgrad_optimizer.make_optimizer(prompt_variable)
        best_prompt = initial_prompt
        best_score = -1.0
        best_iteration = -1
        best_rank = None

        for iteration in range(int(self.experiment["n_optimization_steps"])):
            prompt = self._textgrad_value(prompt_variable)
            image_path = self._optimization_image_path(target_class, iteration)
            seed = self._optimization_seed(iteration)

            self.image_generator.generate(prompt, image_path, seed=seed)
            classifier_result = self.classifier.predict(
                image_path=image_path,
                target_class=target_class,
                top_k=int(self.experiment.get("top_k", 5)),
            )

            current_score = float(classifier_result["target_confidence"])
            if current_score > best_score:
                best_score = current_score
                best_prompt = prompt
                best_iteration = iteration
                best_rank = classifier_result["target_rank"]

            step_result = self.textgrad_optimizer.step(
                prompt_variable,
                optimizer,
                target_class,
                classifier_result,
            )

            self._log_optimization_step(
                target_class,
                iteration,
                prompt,
                step_result,
                seed,
                image_path,
                classifier_result,
            )

        self.best_prompt_metadata[target_class] = {
            "best_iteration": best_iteration,
            "best_target_confidence": best_score,
            "best_target_rank": best_rank,
        }
        return best_prompt

    def _make_initial_prompt(self, target_class: str) -> str:
        """Create a name-free cue prompt for the target class.

        Args:
            target_class: Hidden classifier target, not included in the prompt.

        Returns:
            Initial image-generation prompt.
        """
        # These seed prompts follow prompts/initial_prompt_system.txt: they are
        # name-free and use visual cue families instead of exact class labels.
        prompts = {
            "tabby": (
                "Warm indoor scene with orange-black striped soft textures, plush "
                "curled form, domestic furniture, shallow depth of field, amber daylight."
            ),
            "sports car": (
                "Low glossy aerodynamic silhouette, racetrack asphalt, metallic red "
                "reflections, carbon-fiber texture, showroom lighting, sharp "
                "motion-oriented composition."
            ),
            "cowboy hat": (
                "Dusty western street, wide curved leather brim silhouette, saloon wood, "
                "horse saddle, desert cactus, warm sunset, weathered brown material."
            ),
            "volcano": (
                "Towering mountain peak, dark ash cloud, smoky haze, black basalt rock, "
                "glowing orange cracks, dramatic rugged landscape."
            ),
            "book jacket": (
                "Upright rectangular printed panel, glossy paper texture, bold "
                "typography-like blocks, colorful geometric layout, display table, "
                "studio lighting."
            ),
        }
        return prompts.get(
            target_class,
            "Realistic photographic scene with distinctive textures, materials, colors, "
            "shapes, background context, and co-occurring visual cues.",
        )

    def run_inference(self) -> dict[str, float]:
        """Generate images from final prompts and evaluate activation rate.

        Returns:
            Mapping from target class to activation maximization rate.
        """
        self._reset_inference_logs()
        final_prompts = self._load_final_prompts()
        activation_rates = {}
        inference_summary = {}

        for target_class, prompt in final_prompts.items():
            print(f"Running inference for: {target_class}")
            class_results = []
            n_images = int(self.experiment["n_inference_images"])

            for image_index in range(n_images):
                image_path = self._inference_image_path(target_class, image_index)
                seed = self._inference_seed(image_index)
                self.image_generator.generate(prompt, image_path, seed=seed)

                classifier_result = self.classifier.predict(
                    image_path=image_path,
                    target_class=target_class,
                    top_k=int(self.experiment.get("top_k", 5)),
                )

                top1_correct = classifier_result["target_rank"] == 1
                top5_correct = (
                    classifier_result["target_rank"] is not None
                    and classifier_result["target_rank"] <= 5
                )
                class_results.append(classifier_result)
                self._log_inference_result(
                    target_class,
                    image_index,
                    prompt,
                    seed,
                    image_path,
                    classifier_result,
                    top1_correct,
                    top5_correct,
                )

            summary = self._summarize_inference_results(class_results)
            inference_summary[target_class] = summary
            activation_rates[target_class] = summary["top1_activation_rate"]

        write_json(Path(self.paths["results_dir"]) / "activation_rates.json", activation_rates)
        write_json(Path(self.paths["results_dir"]) / "inference_summary.json", inference_summary)
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
        step_result: dict,
        seed: int,
        image_path: Path,
        classifier_result: dict,
    ) -> None:
        """Save one optimization step to CSV and JSONL."""
        row = {
            "target_class": target_class,
            "iteration": iteration,
            "prompt": prompt,
            "textual_loss": step_result["textual_loss"],
            "candidate_prompt": step_result["candidate_prompt"],
            "accepted_prompt": step_result["accepted_prompt"],
            "forbidden_terms": step_result["forbidden_terms"],
            "was_rejected": step_result["was_rejected"],
            "seed": seed,
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
        seed: int,
        image_path: Path,
        classifier_result: dict,
        top1_correct: bool,
        top5_correct: bool,
    ) -> None:
        """Save one inference image result to CSV."""
        row = {
            "target_class": target_class,
            "image_index": image_index,
            "prompt": prompt,
            "seed": seed,
            "image_path": str(image_path),
            "top1_label": classifier_result["top1_label"],
            "top1_confidence": classifier_result["top1_confidence"],
            "target_confidence": classifier_result["target_confidence"],
            "target_rank": classifier_result["target_rank"],
            "top1_correct": top1_correct,
            "top5_correct": top5_correct,
        }

        append_csv_row(Path(self.paths["results_dir"]) / "inference_results.csv", row, INFERENCE_COLUMNS)

    def _save_final_prompts(self, final_prompts: dict[str, str]) -> None:
        """Save final prompts after optimization."""
        cleaned_prompts = {
            target_class: self.textgrad_optimizer.clean_final_prompt(prompt)
            for target_class, prompt in final_prompts.items()
        }
        results_dir = Path(self.paths["results_dir"])
        write_json(results_dir / "final_prompts.json", cleaned_prompts)
        write_json(results_dir / "best_prompt_metadata.json", self.best_prompt_metadata)

    def _reset_optimization_logs(self) -> None:
        """Remove old optimization outputs before a new run."""
        results_dir = Path(self.paths["results_dir"])
        for file_name in [
            "final_prompts.json",
            "best_prompt_metadata.json",
            "optimization_logs.csv",
            "optimization_logs.jsonl",
        ]:
            path = results_dir / file_name
            if path.exists():
                path.unlink()

    def _reset_inference_logs(self) -> None:
        """Remove old inference outputs before a new run."""
        results_dir = Path(self.paths["results_dir"])
        for file_name in [
            "inference_results.csv",
            "activation_rates.json",
            "inference_summary.json",
            "inference_metrics.json",
        ]:
            path = results_dir / file_name
            if path.exists():
                path.unlink()

    def _summarize_inference_results(self, results: list[dict]) -> dict:
        """Compute class-level inference metrics from classifier outputs."""
        if not results:
            return {
                "top1_activation_rate": 0.0,
                "top5_activation_rate": 0.0,
                "mean_target_confidence": 0.0,
                "median_target_confidence": 0.0,
                "mean_target_rank": None,
            }

        top1_hits = [result["target_rank"] == 1 for result in results]
        top5_hits = [
            result["target_rank"] is not None and result["target_rank"] <= 5
            for result in results
        ]
        confidences = [float(result["target_confidence"]) for result in results]
        valid_ranks = [
            int(result["target_rank"])
            for result in results
            if result["target_rank"] is not None
        ]

        return {
            "top1_activation_rate": sum(top1_hits) / len(results),
            "top5_activation_rate": sum(top5_hits) / len(results),
            "mean_target_confidence": statistics.mean(confidences),
            "median_target_confidence": statistics.median(confidences),
            "mean_target_rank": statistics.mean(valid_ranks) if valid_ranks else None,
        }

    def _load_final_prompts(self) -> dict[str, str]:
        """Load prompts saved by run_optimization()."""
        path = Path(self.paths["results_dir"]) / "final_prompts.json"
        if not path.exists():
            raise FileNotFoundError("Run optimization first: results/final_prompts.json is missing.")

        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _optimization_seed(self, iteration: int) -> int:
        """Return a reproducible seed for one optimization image."""
        base_seed = int(self.config["image_generator"].get("base_seed", 42))
        return base_seed + iteration

    def _inference_seed(self, image_index: int) -> int:
        """Return a reproducible, varying seed for one inference image."""
        base_seed = int(self.config["image_generator"].get("base_seed", 42))
        if not bool(self.config["image_generator"].get("vary_seed", True)):
            return base_seed
        return base_seed + 1000 + image_index

    def _textgrad_value(self, variable) -> str:
        """Read a TextGrad variable value across TextGrad versions."""
        if hasattr(variable, "value"):
            return str(variable.value)
        if hasattr(variable, "get_value"):
            return str(variable.get_value())
        return str(variable)
