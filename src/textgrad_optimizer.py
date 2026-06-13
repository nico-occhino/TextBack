"""TextGrad prompt optimizer for TextBack.

This module contains the real textual-backward backend.  It uses the official
TextGrad API with a LiteLLM backward engine configured from YAML.
"""

import os
from pathlib import Path
import time


FORBIDDEN_TERMS = {
    "tabby": ["tabby", "cat", "kitten", "feline"],
    "sports car": ["sports car", "sport car", "car", "vehicle", "automobile"],
    "cowboy hat": ["cowboy hat"],
    "volcano": ["volcano"],
    "book jacket": ["book jacket", "book"],
}


def clean_prompt(prompt: str, max_prompt_words: int = 60) -> str:
    """Normalize and cap an image-generation prompt."""
    words = prompt.strip().replace("\n", " ").split()
    cleaned = " ".join(words[:max_prompt_words])
    return cleaned.rstrip(" ,;")


def contains_forbidden_terms(prompt: str, target_class: str) -> list[str]:
    """Find target-leaking terms in a candidate prompt."""
    prompt_lower = prompt.lower()
    return [
        term
        for term in FORBIDDEN_TERMS.get(target_class, [])
        if term.lower() in prompt_lower
    ]


class TextGradPromptOptimizer:
    """Optimize image prompts with TextGrad textual gradients."""

    def __init__(self, config: dict) -> None:
        """Configure TextGrad and store optimizer settings."""
        self._load_env_file()
        self.backward_engine = config["textgrad"]["backward_engine"]
        self.cache = bool(config["textgrad"].get("cache", True))
        self.sleep_seconds_after_step = int(config["textgrad"].get("sleep_seconds_after_step", 20))
        self.max_retries_on_rate_limit = int(config["textgrad"].get("max_retries_on_rate_limit", 3))
        self.max_prompt_words = int(config["textgrad"].get("max_prompt_words", 60))
        self.initial_prompt_temperature = float(
            config["textgrad"].get("initial_prompt_temperature", 0.2)
        )
        self.initial_prompt_max_retries = int(
            config["textgrad"].get("initial_prompt_max_retries", 3)
        )
        self.fallback_on_initial_prompt_failure = bool(
            config["textgrad"].get("fallback_on_initial_prompt_failure", True)
        )
        self.prompts_dir = Path(config["paths"].get("prompts_dir", "prompts"))
        self.initial_prompt_system = self._load_prompt_file("initial_prompt_system.txt")
        self.refinement_prompt_system = self._load_prompt_file("refinement_prompt_system.txt")
        self._check_api_key()

        import textgrad as tg

        self.tg = tg

        self._set_backward_engine()
        self.loss_instruction_variable = self.tg.Variable(
            "Initial TextBack loss instruction.",
            requires_grad=False,
            role_description=(
                "non-trainable textual loss instruction that tells TextGrad how to "
                "evaluate and critique the current image-generation prompt using "
                "classifier feedback, target-rank information, and positive descriptors"
            ),
        )
        self.loss_fn = self.tg.TextLoss(self.loss_instruction_variable)

    def make_prompt_variable(self, initial_prompt: str, target_class: str):
        """Create the trainable TextGrad prompt variable."""
        return self.tg.Variable(
            initial_prompt,
            requires_grad=True,
            role_description=(
                "name-free text-to-image prompt optimized through textual feedback "
                f"to activate the hidden visual category '{target_class}'"
            ),
        )

    def make_optimizer(self, prompt_variable):
        """Create one persistent TGD optimizer for a prompt variable."""
        return self.tg.TGD(parameters=[prompt_variable])

    def clean_final_prompt(self, prompt: str) -> str:
        """Clean and cap a prompt before saving it."""
        return clean_prompt(prompt, self.max_prompt_words)

    def generate_initial_prompt(self, target_class: str) -> dict:
        """Generate one name-free initial prompt with the configured LLM."""
        try:
            from litellm import completion
        except ImportError as error:
            raise RuntimeError(
                "litellm is required for LLM initial prompt generation."
            ) from error

        max_retries = max(1, self.initial_prompt_max_retries)
        forbidden_terms_config = FORBIDDEN_TERMS.get(target_class, [])
        last_prompt = ""
        last_forbidden_terms = []
        last_error = ""

        for attempt in range(1, max_retries + 1):
            user_message = (
                f"Target class: {target_class}\n"
                f"Forbidden terms: {forbidden_terms_config}\n"
                f"Attempt: {attempt}/{max_retries}\n"
                "Generate a name-free prompt.\n"
                "Do not use any forbidden terms.\n"
                "Return only the prompt.\n"
                f"Maximum {self.max_prompt_words} words."
            )
            try:
                response = completion(
                    model=self._litellm_model_name(),
                    messages=[
                        {"role": "system", "content": self.initial_prompt_system},
                        {"role": "user", "content": user_message},
                    ],
                    temperature=self.initial_prompt_temperature,
                )
                prompt = clean_prompt(self._completion_text(response), self.max_prompt_words)
            except Exception as error:
                self._print_textgrad_failure(error)
                last_error = str(error)
                if attempt == max_retries and not self.fallback_on_initial_prompt_failure:
                    raise RuntimeError(
                        "Initial LLM prompt generation failed and fallback is disabled."
                    ) from error
                print(
                    f"Initial prompt attempt {attempt}/{max_retries} failed for "
                    f"{target_class}: {error}"
                )
                continue

            forbidden_terms = contains_forbidden_terms(prompt, target_class)
            if not forbidden_terms:
                return {
                    "prompt": prompt,
                    "source": "llm_generated",
                    "attempts": attempt,
                    "forbidden_terms": [],
                }

            last_prompt = prompt
            last_forbidden_terms = forbidden_terms
            forbidden_text = ", ".join(forbidden_terms)
            print(
                f"Initial prompt attempt {attempt}/{max_retries} leaked forbidden "
                f"terms for {target_class}: {forbidden_text}"
            )

        return {
            "prompt": "",
            "source": "llm_failed",
            "attempts": max_retries,
            "forbidden_terms": last_forbidden_terms,
            "error": last_error
            or f"Forbidden terms leaked after retries. Last prompt: {last_prompt}",
        }

    def build_loss_instruction(
        self,
        target_class: str,
        classifier_result: dict,
        positive_descriptors: list[str] | None = None,
    ) -> str:
        """Build a natural-language loss from classifier feedback."""
        topk_lines = []
        for prediction in classifier_result["topk"]:
            topk_lines.append(
                f"- {prediction['index']}: {prediction['label']} "
                f"({prediction['confidence']:.4f})"
            )

        descriptor_section = ""
        if positive_descriptors:
            visible_descriptors = positive_descriptors[:5]
            descriptor_lines = "\n".join(
                f"- {descriptor}" for descriptor in visible_descriptors
            )
            descriptor_section = (
                "\n\nPositive descriptors to preserve:\n"
                f"{descriptor_lines}\n"
                "Preserve these if useful and allowed."
            )

        return (
            self.refinement_prompt_system
            + "\n\nLive classifier feedback:\n"
            f"Target class: {target_class}\n"
            f"Top-1 prediction: {classifier_result['top1_label']} "
            f"with confidence {classifier_result['top1_confidence']:.4f}\n"
            f"Target confidence: {classifier_result['target_confidence']:.4f}\n"
            f"Target rank: {classifier_result['target_rank']}\n"
            "Top-k predictions:\n"
            + "\n".join(topk_lines)
            + descriptor_section
            + "\nConstraint reminder: The improved image-generation prompt must not "
            "contain the exact target class name or close synonyms.\n"
            f"Maximum word reminder: return at most {self.max_prompt_words} words."
        )

    def step(
        self,
        prompt_variable,
        optimizer,
        target_class: str,
        classifier_result: dict,
        positive_descriptors: list[str] | None = None,
    ) -> dict:
        """Run one TextGrad optimization step."""
        loss_instruction = self.build_loss_instruction(
            target_class,
            classifier_result,
            positive_descriptors=positive_descriptors,
        )
        previous_prompt = clean_prompt(self._value_of(prompt_variable), self.max_prompt_words)

        self._set_value(self.loss_instruction_variable, loss_instruction)
        try:
            loss = self._run_textgrad_update_with_retries(self.loss_fn, prompt_variable, optimizer)
        except (IndexError, RuntimeError) as error:
            if not self._is_textgrad_format_error(error):
                raise
            self._set_value(prompt_variable, previous_prompt)
            return {
                "accepted_prompt": previous_prompt,
                "candidate_prompt": previous_prompt,
                "textual_loss": f"[REJECTED textgrad_format_error] {error}",
                "forbidden_terms": "",
                "was_rejected": True,
            }

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

                loss.backward()

                optimizer.step()

                time.sleep(self.sleep_seconds_after_step)
                return loss
            except Exception as error:
                self._print_textgrad_failure(error)
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
        error_type = type(error).__name__.lower()
        message = str(error).lower()
        rate_limit_markers = [
            "ratelimit",
            "429",
            "rate limit",
            "too many requests",
            "quota",
        ]
        return any(
            marker in error_type or marker in message
            for marker in rate_limit_markers
        )

    def _print_textgrad_failure(self, error: Exception) -> None:
        """Print concise TextGrad/LiteLLM failure details without secrets."""
        message = str(error).replace("\n", " ")[:500]
        print(f"TextGrad/LiteLLM failure type: {type(error).__name__}")
        print(f"TextGrad/LiteLLM failure message: {message}")

    def _is_textgrad_format_error(self, error: Exception) -> bool:
        """Return True when TextGrad cannot parse an optimizer response."""
        if isinstance(error, IndexError):
            return True
        message = str(error).lower()
        format_error_markers = [
            "optimizer response could not be indexed",
            "could not be indexed",
            "new_variable_tags",
        ]
        return any(marker in message for marker in format_error_markers)

    def _set_backward_engine(self) -> None:
        """Set TextGrad's global backward engine with a small compatibility shim."""
        try:
            self.tg.set_backward_engine(self.backward_engine, override=True, cache=self.cache)
        except TypeError:
            self.tg.set_backward_engine(self.backward_engine, override=True)

    def _check_api_key(self) -> None:
        """Check the API key expected by the selected TextGrad backend."""
        normalized_backend = self._normalize_backend_name(self.backward_engine)
        provider = self._backend_provider(normalized_backend)

        print(f"TextGrad backend: {self.backward_engine}")
        print(f"Detected provider: {provider or 'unknown'}")

        if provider != "openai":
            raise RuntimeError(
                "Only OpenAI TextGrad backends are supported in this student version."
            )

        has_key = bool(os.getenv("OPENAI_API_KEY"))
        print(f"OPENAI_API_KEY found: {has_key}")
        if not has_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI TextGrad backend.")

    def _normalize_backend_name(self, backend: str) -> str:
        """Normalize TextGrad backend names for provider checks."""
        backend = backend.strip().lower()
        if backend.startswith("experimental:"):
            backend = backend.replace("experimental:", "", 1)
        if backend.startswith("experimental/"):
            backend = backend.replace("experimental/", "", 1)
        if backend.startswith("gpt-"):
            backend = f"openai/{backend}"
        return backend

    def _backend_provider(self, normalized_backend: str) -> str | None:
        """Return the provider encoded by a normalized backend string."""
        if "openai" in normalized_backend:
            return "openai"
        return None

    def _load_env_file(self) -> None:
        """Load .env with python-dotenv when the package is installed."""
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass

    def _load_prompt_file(self, file_name: str) -> str:
        """Read one prompt file from the configured prompts directory."""
        path = self.prompts_dir / file_name
        if not path.exists():
            display_path = (self.prompts_dir / file_name).as_posix()
            raise RuntimeError(f"Missing prompt file: {display_path}")

        return path.read_text(encoding="utf-8").strip()

    def _litellm_model_name(self) -> str:
        """Return the LiteLLM model name for the configured TextGrad backend."""
        return self._normalize_backend_name(self.backward_engine)

    def _completion_text(self, response) -> str:
        """Extract assistant text from a LiteLLM completion response."""
        try:
            return str(response["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError):
            pass

        try:
            return str(response.choices[0].message.content)
        except (AttributeError, IndexError, TypeError) as error:
            raise RuntimeError("Could not read text from LiteLLM response.") from error

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
