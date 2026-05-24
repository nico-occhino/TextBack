"""TextGrad prompt optimizer for TextBack.

This module contains the real textual-backward backend.  It uses the official
TextGrad API with a LiteLLM backward engine configured from YAML.
"""

import os


def clean_prompt(prompt: str) -> str:
    """Normalize an image-generation prompt without shortening it.

    Args:
        prompt: Raw prompt text returned by TextGrad or another LLM.

    Returns:
        Prompt with whitespace cleaned.
    """
    return " ".join(prompt.strip().replace("\n", " ").split())


class TextGradPromptOptimizer:
    """Optimize image prompts with TextGrad textual gradients."""

    def __init__(self, config: dict) -> None:
        """Configure TextGrad and store optimizer settings.

        Args:
            config: Loaded project configuration.
        """
        self._load_env_file()
        self.backward_engine = config["textgrad"]["backward_engine"]
        self.cache = bool(config["textgrad"].get("cache", True))
        self._check_api_key()

        import textgrad as tg

        self.tg = tg

        # TextGrad uses this engine during loss.backward(), where textual
        # gradients are produced by the LLM.
        self._set_backward_engine()

    def make_prompt_variable(self, initial_prompt: str, target_class: str):
        """Create the optimizable TextGrad prompt variable.

        Args:
            initial_prompt: Starting text-to-image prompt.
            target_class: ImageNet class we want to activate.

        Returns:
            A TextGrad Variable whose text value can be optimized.
        """
        # tg.Variable is the optimizable prompt, similar to a torch tensor with
        # requires_grad=True, but for natural language.
        return self.tg.Variable(
            initial_prompt,
            requires_grad=True,
            role_description=(
                "image-generation prompt optimized to activate ImageNet class "
                f"'{target_class}'"
            ),
        )

    def build_loss_instruction(self, target_class: str, classifier_result: dict) -> str:
        """Build a natural-language loss from classifier feedback.

        Args:
            target_class: Desired ImageNet class.
            classifier_result: Output dictionary from classifier.predict().

        Returns:
            Instruction string passed to tg.TextLoss.
        """
        topk_lines = []
        for prediction in classifier_result["topk"]:
            topk_lines.append(
                f"- {prediction['index']}: {prediction['label']} "
                f"({prediction['confidence']:.4f})"
            )

        return (
            "You are optimizing a text-to-image prompt for shortcut discovery.\n"
            f"Target class: {target_class}\n"
            f"Top-1 prediction: {classifier_result['top1_label']} "
            f"with confidence {classifier_result['top1_confidence']:.4f}\n"
            f"Target confidence: {classifier_result['target_confidence']:.4f}\n"
            f"Target rank: {classifier_result['target_rank']}\n"
            "Top-k predictions:\n"
            + "\n".join(topk_lines)
            + "\nThe improved image-generation prompt must NOT mention the target class, "
            "direct object names, or direct object parts. Search for indirect shortcut "
            "cues: background, texture, color, lighting, scene context, co-occurring "
            "objects, shapes, and materials that may spuriously activate the classifier. "
            "Return only the improved image-generation prompt."
        )

    def step(self, prompt_variable, target_class: str, classifier_result: dict) -> tuple[str, str]:
        """Run one TextGrad optimization step.

        Args:
            prompt_variable: TextGrad Variable containing the current prompt.
            target_class: Desired ImageNet class.
            classifier_result: Output dictionary from classifier.predict().

        Returns:
            Updated prompt text and textual loss/feedback string.
        """
        loss_instruction = self.build_loss_instruction(target_class, classifier_result)

        # tg.TextLoss turns classifier feedback into a textual loss function.
        loss_fn = self.tg.TextLoss(loss_instruction)

        # TGD is Textual Gradient Descent: it updates text using gradients.
        optimizer = self.tg.TGD(parameters=[prompt_variable])
        optimizer.zero_grad()

        loss = loss_fn(prompt_variable)

        # loss.backward() asks the backward engine for textual gradients.
        loss.backward()

        # optimizer.step() applies those textual gradients to the prompt.
        optimizer.step()

        cleaned_prompt = clean_prompt(self._value_of(prompt_variable))
        self._set_value(prompt_variable, cleaned_prompt)
        return cleaned_prompt, self._value_of(loss)

    def _set_backward_engine(self) -> None:
        """Set TextGrad's global backward engine with a small compatibility shim."""
        try:
            self.tg.set_backward_engine(self.backward_engine, override=True, cache=self.cache)
        except TypeError:
            # Older TextGrad versions may not expose the cache argument here.
            self.tg.set_backward_engine(self.backward_engine, override=True)

    def _check_api_key(self) -> None:
        """Check the API key expected by the selected TextGrad backend."""
        backend = self.backward_engine.lower()

        if "gemini" in backend and not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
            raise RuntimeError(
                f"Missing API key for selected TextGrad backend: {self.backward_engine}. "
                "Add it to .env."
            )

        if "groq" in backend and not os.getenv("GROQ_API_KEY"):
            raise RuntimeError(
                f"Missing API key for selected TextGrad backend: {self.backward_engine}. "
                "Add it to .env."
            )

        if "gemini" not in backend and "groq" not in backend:
            print(
                f"Warning: no provider-specific API key check for TextGrad backend: "
                f"{self.backward_engine}"
            )

    def _load_env_file(self) -> None:
        """Load .env with python-dotenv when the package is installed."""
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            # The real environment should install python-dotenv.  This fallback
            # still allows API keys provided by the shell to work.
            pass

    def _value_of(self, variable) -> str:
        """Read the text value from a TextGrad object across versions."""
        if hasattr(variable, "value"):
            return str(variable.value)
        if hasattr(variable, "get_value"):
            return str(variable.get_value())
        return str(variable)

    def _set_value(self, variable, value: str) -> None:
        """Write a cleaned value back to a TextGrad variable."""
        if hasattr(variable, "set_value"):
            variable.set_value(value)
        else:
            variable.value = value
