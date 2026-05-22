"""Image generator interfaces.

The dummy generator creates placeholder images with the prompt written on them.
This validates data flow before connecting a diffusion model or an API.
"""

from pathlib import Path
from textwrap import wrap
from typing import Dict

from PIL import Image, ImageDraw


class BaseImageGenerator:
    """Minimal interface for text-to-image generation."""

    def generate(self, prompt: str, output_path: str | Path) -> Path:
        """Generate an image from a prompt.

        Args:
            prompt: Text prompt sent to the generator.
            output_path: File path where the image should be saved.

        Returns:
            Path to the saved image.
        """
        raise NotImplementedError


class DummyImageGenerator(BaseImageGenerator):
    """Placeholder image generator used for dry runs."""

    def __init__(self, width: int = 512, height: int = 512) -> None:
        """Store image dimensions.

        Args:
            width: Output image width in pixels.
            height: Output image height in pixels.
        """
        self.width = width
        self.height = height

    def generate(self, prompt: str, output_path: str | Path) -> Path:
        """Create a simple image that contains the prompt text.

        Args:
            prompt: Text prompt to display on the placeholder image.
            output_path: File path where the image should be saved.

        Returns:
            Path to the saved placeholder image.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Use a stable background color so identical prompts are recognizable.
        color = self._color_from_prompt(prompt)
        image = Image.new("RGB", (self.width, self.height), color=color)
        draw = ImageDraw.Draw(image)

        draw.rectangle((24, 24, self.width - 24, self.height - 24), outline="white", width=3)
        draw.text((40, 40), "TextBack dummy image", fill="white")

        y = 90
        for line in wrap(prompt, width=48):
            draw.text((40, y), line, fill="white")
            y += 24

        image.save(output_path)
        return output_path

    def _color_from_prompt(self, prompt: str) -> tuple[int, int, int]:
        """Create a deterministic RGB color from prompt text.

        Args:
            prompt: Input prompt.

        Returns:
            RGB color tuple.
        """
        value = abs(hash(prompt))
        return (60 + value % 120, 60 + (value // 7) % 120, 60 + (value // 13) % 120)


def build_image_generator(config: Dict) -> BaseImageGenerator:
    """Create the image generator requested by the config.

    Args:
        config: Loaded configuration dictionary.

    Returns:
        An object implementing BaseImageGenerator.
    """
    provider = config.get("image_generator", {}).get("provider", "dummy")
    if provider != "dummy":
        print("Only the dummy image generator is implemented; falling back to dummy.")

    gen_config = config.get("image_generator", {})
    return DummyImageGenerator(
        width=int(gen_config.get("width", 512)),
        height=int(gen_config.get("height", 512)),
    )
