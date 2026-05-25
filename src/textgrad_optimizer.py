"""TextGrad prompt optimizer for TextBack.

This module contains the real textual-backward backend.  It uses the official
TextGrad API with a LiteLLM backward engine configured from YAML.
"""

import os
import time


FORBIDDEN_TERMS = {
    "tabby": ["tabby", "cat", "kitten", "feline", "whisker", "paw", "tail"],
    "sports car": ["sports car", "sport car", "car", "vehicle", "wheel", "tire", "hood", "headlight"],
    "cowboy hat": ["cowboy hat", "hat", "brim", "crown"],
    "volcano": ["volcano", "lava", "crater", "eruption"],
    "book jacket": ["book jacket", "book", "cover", "spine", "title"],
}


def clean_prompt(prompt: str, max_prompt_words: int = 55) -> str:
    """Normalize and cap an image-generation prompt.

    Args:
        prompt: Raw prompt text returned by TextGrad or another LLM.
        max_prompt_words: Maximum number of words kept for Stable Diffusion.

    Returns:
        Prompt with whitespace cleaned and capped by word count.
    """
    words = prompt.strip().replace("\n", " ").split()
    return " ".join(words[:max_prompt_words])


def contains_forbidden_terms(prompt: str, target_class: str) -> list[str]:
    """Find target-leaking terms in a candidate shortcut prompt.

    Args:
        prompt: Candidate image-generation prompt.
        target_class: Hidden ImageNet target class.

    Returns:
        List of forbidden terms found in the prompt.
    """
    prompt_lower = prompt.lower()
    return [
        term
        for term in FORBIDDEN_TERMS.get(target_class, [])
        if term.lower() in prompt_lower
    ]


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
        self.sleep_seconds_after_step = int(config["textgrad"].get("sleep_seconds_after_step", 20))
        self.max_retries_on_rate_limit = int(config["textgrad"].get("max_retries_on_rate_limit", 3))
        self.max_prompt_words = int(config["textgrad"].get("max_prompt_words", 55))
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

    def make_optimizer(self, prompt_variable):
        """Create one persistent TGD optimizer for a prompt variable."""
        return self.tg.TGD(parameters=[prompt_variable])

    def clean_final_prompt(self, prompt: str) -> str:
        """Clean and cap a prompt before saving it."""
        return clean_prompt(prompt, self.max_prompt_words)

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

    def step(self, prompt_variable, optimizer, target_class: str, classifier_result: dict) -> tuple[str, str]:
        """Run one TextGrad optimization step.

        Args:
            prompt_variable: TextGrad Variable containing the current prompt.
            optimizer: Persistent TextGrad TGD optimizer for this prompt.
            target_class: Desired ImageNet class.
            classifier_result: Output dictionary from classifier.predict().

        Returns:
            Updated prompt text and textual loss/feedback string.
        """
        loss_instruction = self.build_loss_instruction(target_class, classifier_result)
        previous_prompt = clean_prompt(self._value_of(prompt_variable), self.max_prompt_words)

        # tg.TextLoss turns classifier feedback into a textual loss function.
        loss_fn = self.tg.TextLoss(loss_instruction)

        loss = self._run_textgrad_update_with_retries(loss_fn, prompt_variable, optimizer)

        candidate_prompt = clean_prompt(self._value_of(prompt_variable), self.max_prompt_words)
        textual_loss = self._value_of(loss)
        forbidden_terms = contains_forbidden_terms(candidate_prompt, target_class)
        if forbidden_terms:
            self._set_value(prompt_variable, previous_prompt)
            forbidden_text = ", ".join(forbidden_terms)
            return {
                "accepted_prompt": previous_prompt,
                "candidate_prompt": candidate_prompt,
                "textual_loss": f"{textual_loss} [REJECTED forbidden_terms={forbidden_text}]",
                "forbidden_terms": forbidden_text,
                "was_rejected": True,
            }

        self._set_value(prompt_variable, candidate_prompt)
        return {
            "accepted_prompt": candidate_prompt,
            "candidate_prompt": candidate_prompt,
            "textual_loss": textual_loss,
            "forbidden_terms": "",
            "was_rejected": False,
        }

    def _run_textgrad_update_with_retries(self, loss_fn, prompt_variable, optimizer):
        """Run loss/backward/step with simple rate-limit retries."""
        max_attempts = self.max_retries_on_rate_limit + 1

        for attempt in range(1, max_attempts + 1):
            try:
                optimizer.zero_grad()
                loss = loss_fn(prompt_variable)

                # loss.backward() asks the backward engine for textual gradients.
                loss.backward()

                # optimizer.step() applies those textual gradients to the prompt.
                optimizer.step()

                time.sleep(self.sleep_seconds_after_step)
                return loss
            except Exception as error:
                if self._is_rate_limit_error(error) and attempt < max_attempts:
                    print(
                        "Rate limit reached during TextGrad step. "
                        f"Sleeping {self.sleep_seconds_after_step} seconds before retry "
                        f"{attempt}/{self.max_retries_on_rate_limit}."
                    )
                    time.sleep(self.sleep_seconds_after_step)
                    continue
                if self._is_rate_limit_error(error):
                    raise RuntimeError(
                        "TextGrad step failed after rate-limit retries. Increase "
                        "sleep_seconds_after_step or reduce classes per run."
                    ) from error
                raise

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Return True when an exception looks like a provider rate limit."""
        message = str(error).lower()
        rate_limit_markers = [
            "429",
            "rate limit",
            "too many requests",
            "tpm",
            "tokens per minute",
            "retryerror",
        ]
        return any(marker in message for marker in rate_limit_markers)

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
