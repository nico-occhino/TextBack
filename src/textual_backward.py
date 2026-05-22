"""Custom TextGrad-like textual backward logic.

TextGrad backpropagates feedback through text.  Our simplified version reads
classifier feedback, turns it into a textual loss, and asks the LLM client for a
better prompt.
"""

from pathlib import Path

from src.llm_client import DummyLLMClient


class TextualBackwardOptimizer:
    """Small prompt optimizer that uses prompt files and an LLM client."""

    def __init__(self, llm_client: DummyLLMClient, prompts_dir: str | Path) -> None:
        """Load system prompt files used by the textual update steps.

        Args:
            llm_client: Client with initial/loss/refinement methods.
            prompts_dir: Directory containing prompt text files.
        """
        self.llm_client = llm_client
        self.prompts_dir = Path(prompts_dir)
        self.initial_system_prompt = self._read_prompt("initial_prompt_system.txt")
        self.loss_system_prompt = self._read_prompt("textual_loss_system.txt")
        self.refinement_system_prompt = self._read_prompt("refinement_prompt_system.txt")

    def initial_prompt(self, target_class: str) -> str:
        """Generate the first prompt for one target class.

        Args:
            target_class: Desired ImageNet class.

        Returns:
            Initial image prompt.
        """
        return self.llm_client.generate_initial_prompt(target_class, self.initial_system_prompt)

    def textual_loss(self, target_class: str, current_prompt: str, classifier_result: dict) -> str:
        """Compute natural language feedback for the current prompt.

        Args:
            target_class: Desired ImageNet class.
            current_prompt: Prompt used for the current generated image.
            classifier_result: Output from classifier.predict().

        Returns:
            Textual loss string.
        """
        return self.llm_client.compute_textual_loss(
            target_class,
            current_prompt,
            classifier_result,
            self.loss_system_prompt,
        )

    def refine(self, target_class: str, current_prompt: str, classifier_result: dict) -> str:
        """Create the next prompt using classifier feedback.

        Args:
            target_class: Desired ImageNet class.
            current_prompt: Prompt used for the current generated image.
            classifier_result: Output from classifier.predict().

        Returns:
            Refined prompt.
        """
        return self.llm_client.refine_prompt(
            target_class,
            current_prompt,
            classifier_result,
            self.refinement_system_prompt,
        )

    def _read_prompt(self, file_name: str) -> str:
        """Read one prompt file and return a fallback if it is empty.

        Args:
            file_name: Prompt file name inside prompts_dir.

        Returns:
            Prompt text.
        """
        path = self.prompts_dir / file_name
        if not path.exists():
            return "Use concise, practical instructions."

        text = path.read_text(encoding="utf-8").strip()
        return text or "Use concise, practical instructions."
