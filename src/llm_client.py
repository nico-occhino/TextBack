"""Dummy LLM client used by the first TextBack version.

The real project can later replace this class with an API client.  For now the
methods return deterministic text, which lets the whole pipeline run offline.
"""


class DummyLLMClient:
    """Small deterministic replacement for a language model."""

    def generate_initial_prompt(self, target_class: str, system_prompt: str) -> str:
        """Create the first image prompt for a target class.

        Args:
            target_class: ImageNet class we want the classifier to activate.
            system_prompt: Instruction text loaded from prompts/.

        Returns:
            A simple text-to-image prompt.
        """
        return (
            f"A clear ImageNet-style photo of a {target_class}. "
            f"The {target_class} is centered, well lit, and easy to recognize."
        )

    def compute_textual_loss(
        self,
        target_class: str,
        current_prompt: str,
        classifier_feedback: dict,
        system_prompt: str,
    ) -> str:
        """Describe what went wrong in natural language.

        Args:
            target_class: Desired ImageNet class.
            current_prompt: Prompt that produced the current image.
            classifier_feedback: Dictionary returned by the classifier.
            system_prompt: Instruction text loaded from prompts/.

        Returns:
            Textual loss/feedback for the current prompt.
        """
        target_rank = classifier_feedback.get("target_rank")
        top1_label = classifier_feedback.get("top1_label", "unknown")
        target_confidence = classifier_feedback.get("target_confidence", 0.0)

        if target_rank == 1:
            return (
                f"The prompt is successful: the classifier predicts {target_class} "
                f"as top-1 with confidence {target_confidence:.3f}."
            )

        return (
            f"The classifier prefers {top1_label} instead of {target_class}. "
            f"The target confidence is {target_confidence:.3f}. "
            f"The next prompt should emphasize canonical {target_class} features "
            f"and remove cues related to {top1_label}."
        )

    def refine_prompt(
        self,
        target_class: str,
        current_prompt: str,
        classifier_feedback: dict,
        system_prompt: str,
    ) -> str:
        """Update the image prompt using classifier feedback.

        Args:
            target_class: Desired ImageNet class.
            current_prompt: Prompt that produced the current image.
            classifier_feedback: Dictionary returned by the classifier.
            system_prompt: Instruction text loaded from prompts/.

        Returns:
            Refined prompt for the next optimization step.
        """
        top1_label = classifier_feedback.get("top1_label", "unknown")

        if classifier_feedback.get("target_rank") == 1:
            update = f"Keep only the {target_class} as the main object."
        else:
            update = f"Make it less like {top1_label} and more clearly like {target_class}."

        return (
            f"{current_prompt} {update} "
            f"Use a plain background and avoid distracting context."
        )
