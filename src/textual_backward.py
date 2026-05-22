"""Textual backward logic.

This module contains the prompt-level optimization pieces inspired by TextGrad:
generate a prompt, read classifier feedback, and ask the LLM to improve the
prompt in natural language.
"""

from typing import Dict

from src.llm_client import BaseLLMClient


def generate_initial_prompt(llm_client: BaseLLMClient, class_name: str) -> str:
    """Generate the first prompt for a target class.

    Args:
        llm_client: LLM client used to write prompts.
        class_name: Target ImageNet class.

    Returns:
        Initial text-to-image prompt.
    """
    return llm_client.generate_initial_prompt(class_name)


def build_classifier_feedback(class_name: str, classifier_output: Dict) -> Dict:
    """Convert classifier output into compact LLM feedback.

    Args:
        class_name: Target ImageNet class.
        classifier_output: Output dictionary returned by the classifier.

    Returns:
        Feedback dictionary used for prompt refinement.
    """
    return {
        "target_class": class_name,
        "target_confidence": classifier_output["target_confidence"],
        "target_rank": classifier_output["target_rank"],
        "top_predictions": classifier_output["top_predictions"],
    }


def refine_prompt(
    llm_client: BaseLLMClient,
    class_name: str,
    current_prompt: str,
    classifier_output: Dict,
    step: int,
) -> str:
    """Refine a prompt using classifier feedback.

    Args:
        llm_client: LLM client used to refine prompts.
        class_name: Target ImageNet class.
        current_prompt: Prompt used for the current generated image.
        classifier_output: Output dictionary returned by the classifier.
        step: Current optimization step.

    Returns:
        Refined prompt for the next step.
    """
    feedback = build_classifier_feedback(class_name, classifier_output)
    return llm_client.refine_prompt(class_name, current_prompt, feedback, step)
