"""LLM client interfaces.

The first version uses a dummy LLM so the whole pipeline runs without paid APIs.
Later, this file is the natural place to add OpenAI, Gemini, or local LLM calls.
"""

from dataclasses import dataclass
from typing import Dict, List


class BaseLLMClient:
    """Minimal interface used by the textual backward pipeline."""

    def generate_initial_prompt(self, class_name: str) -> str:
        """Create the first image prompt for a target class.

        Args:
            class_name: ImageNet class name to visualize.

        Returns:
            A text prompt for the image generator.
        """
        raise NotImplementedError

    def refine_prompt(self, class_name: str, current_prompt: str, feedback: Dict, step: int) -> str:
        """Improve a prompt using classifier feedback.

        Args:
            class_name: Target ImageNet class.
            current_prompt: Prompt used in the previous step.
            feedback: Dictionary containing top predictions and target rank.
            step: Current optimization step.

        Returns:
            A refined text prompt.
        """
        raise NotImplementedError


@dataclass
class DummyLLMClient(BaseLLMClient):
    """Simple deterministic LLM replacement used in dry-run mode."""

    style_words: List[str] | None = None

    def __post_init__(self) -> None:
        """Fill default style words after dataclass initialization."""
        if self.style_words is None:
            self.style_words = ["centered", "well lit", "clear background"]

    def generate_initial_prompt(self, class_name: str) -> str:
        """Return a plain prompt that mentions the target class.

        Args:
            class_name: ImageNet class name to visualize.

        Returns:
            A simple prompt suitable for a dummy or real image generator.
        """
        return f"A photo of a {class_name}, centered, natural colors."

    def refine_prompt(self, class_name: str, current_prompt: str, feedback: Dict, step: int) -> str:
        """Append small, readable refinements based on classifier feedback.

        Args:
            class_name: Target ImageNet class.
            current_prompt: Prompt used in the previous step.
            feedback: Classifier feedback dictionary.
            step: Current optimization step.

        Returns:
            A slightly stronger prompt for the next image.
        """
        target_rank = feedback.get("target_rank")
        top_label = feedback.get("top_predictions", [{}])[0].get("label", "unknown")

        # The dummy refinement imitates TextGrad-style natural language feedback.
        if target_rank == 1:
            note = f"Keep the {class_name} as the only main object."
        else:
            note = f"Make it look less like {top_label} and more clearly like {class_name}."

        return (
            f"{current_prompt} Refinement {step}: {note} "
            f"Use canonical visual features and avoid distracting objects."
        )


def build_llm_client(config: Dict) -> BaseLLMClient:
    """Create the LLM client requested by the config.

    Args:
        config: Loaded configuration dictionary.

    Returns:
        An object implementing BaseLLMClient.
    """
    provider = config.get("llm", {}).get("provider", "dummy")
    if provider != "dummy":
        print("Only the dummy LLM is implemented; falling back to dummy.")
    return DummyLLMClient()
