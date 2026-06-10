"""Descriptor memory helpers for TextBack prompt optimization."""

import re

from src.textgrad_optimizer import contains_forbidden_terms


GENERIC_DESCRIPTORS = {
    "realistic photography",
    "cinematic composition",
    "high quality",
    "image",
    "photo",
    "scene",
    "lighting",
}


def extract_descriptors_from_prompt(
    prompt: str,
    max_descriptor_words: int,
    target_class: str | None = None,
) -> list[str]:
    """Extract short visual descriptors from a prompt.

    Args:
        prompt: Prompt text to split into candidate descriptors.
        max_descriptor_words: Maximum words allowed in one descriptor.
        target_class: Optional target class for forbidden-term filtering.

    Returns:
        Unique descriptors in readable form.
    """
    descriptors = []
    seen = set()
    for fragment in re.split(r"[,;.]", prompt):
        descriptor = " ".join(fragment.strip().split())
        descriptor_key = descriptor.lower()
        if not descriptor or descriptor_key in seen:
            continue
        if descriptor_key in GENERIC_DESCRIPTORS:
            continue
        if len(descriptor.split()) > max_descriptor_words:
            continue
        if target_class and contains_forbidden_terms(descriptor, target_class):
            continue

        descriptors.append(descriptor)
        seen.add(descriptor_key)

    return descriptors


def update_descriptor_memory(
    memory: dict[str, list[str]],
    target_class: str,
    prompt: str,
    classifier_result: dict,
    config: dict,
) -> dict[str, list[str]]:
    """Update positive descriptor memory from one optimization result."""
    memory_config = config.get("descriptor_memory", {})
    if not bool(memory_config.get("enabled", False)):
        return memory

    target_rank = classifier_result.get("target_rank")
    target_confidence = float(classifier_result.get("target_confidence", 0.0))
    rank_threshold = int(memory_config.get("positive_rank_threshold", 5))
    confidence_threshold = float(memory_config.get("min_confidence_threshold", 0.03))
    is_positive = (
        target_rank is not None
        and int(target_rank) <= rank_threshold
    ) or target_confidence >= confidence_threshold
    if not is_positive:
        return memory

    max_descriptor_words = int(memory_config.get("max_descriptor_words", 6))
    max_descriptors = int(memory_config.get("max_descriptors_per_class", 12))
    current_descriptors = memory.setdefault(target_class, [])
    seen = {descriptor.lower() for descriptor in current_descriptors}

    for descriptor in extract_descriptors_from_prompt(
        prompt,
        max_descriptor_words=max_descriptor_words,
        target_class=target_class,
    ):
        descriptor_key = descriptor.lower()
        if descriptor_key in seen:
            continue
        current_descriptors.append(descriptor)
        seen.add(descriptor_key)
        if len(current_descriptors) >= max_descriptors:
            break

    memory[target_class] = current_descriptors[:max_descriptors]
    return memory
