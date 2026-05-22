"""Image generation utilities for TextBack.

The first runnable version uses a dummy generator.  A Stable Diffusion skeleton
is included to show where a real local generator can be added later.
"""

from pathlib import Path
from textwrap import wrap
import struct
import zlib


class DummyImageGenerator:
    """Create simple placeholder images from prompts."""

    def __init__(self, width: int = 512, height: int = 512) -> None:
        """Store output image size.

        Args:
            width: Image width in pixels.
            height: Image height in pixels.
        """
        self.width = width
        self.height = height

    def generate(self, prompt: str, output_path: str | Path) -> Path:
        """Generate a placeholder RGB image and save it.

        Args:
            prompt: Text prompt used to describe the desired image.
            output_path: Path where the image should be saved.

        Returns:
            Path to the saved image.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self._generate_with_pil(prompt, output_path)
        except ModuleNotFoundError:
            # Fallback for bare Python environments.  PIL is better because it
            # can write text, but this still produces a valid RGB PNG.
            self._generate_plain_png(prompt, output_path)

        return output_path

    def _generate_with_pil(self, prompt: str, output_path: Path) -> None:
        """Use PIL to create a readable placeholder image.

        Args:
            prompt: Prompt text to draw on the image.
            output_path: Image path to save.
        """
        from PIL import Image, ImageDraw

        image = Image.new("RGB", (self.width, self.height), self._color_from_prompt(prompt))
        draw = ImageDraw.Draw(image)

        draw.rectangle((24, 24, self.width - 24, self.height - 24), outline="white", width=3)
        draw.text((40, 40), "TextBack dummy image", fill="white")

        y_position = 90
        for line in wrap(prompt, width=52):
            draw.text((40, y_position), line, fill="white")
            y_position += 24

        image.save(output_path)

    def _generate_plain_png(self, prompt: str, output_path: Path) -> None:
        """Write a simple solid-color PNG without external dependencies.

        Args:
            prompt: Prompt used only to choose a deterministic color.
            output_path: Image path to save.
        """
        red, green, blue = self._color_from_prompt(prompt)
        row = b"\x00" + bytes([red, green, blue]) * self.width
        raw_pixels = row * self.height

        def chunk(name: bytes, data: bytes) -> bytes:
            return (
                struct.pack(">I", len(data))
                + name
                + data
                + struct.pack(">I", zlib.crc32(name + data) & 0xFFFFFFFF)
            )

        png = b"\x89PNG\r\n\x1a\n"
        png += chunk("IHDR".encode(), struct.pack(">IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0))
        png += chunk("IDAT".encode(), zlib.compress(raw_pixels))
        png += chunk("IEND".encode(), b"")
        output_path.write_bytes(png)

    def _color_from_prompt(self, prompt: str) -> tuple[int, int, int]:
        """Create a deterministic RGB color from prompt text.

        Args:
            prompt: Text prompt.

        Returns:
            RGB tuple.
        """
        value = abs(hash(prompt))
        return (70 + value % 120, 70 + (value // 7) % 120, 70 + (value // 13) % 120)


class LocalDiffusersGenerator:
    """Skeleton for a future local Stable Diffusion generator.

    This is intentionally not the default.  Later, the constructor can load a
    diffusers pipeline, and generate() can call the pipeline with the prompt.
    """

    def __init__(self, config: dict) -> None:
        """Store config for future Stable Diffusion integration.

        Args:
            config: Loaded project configuration.
        """
        self.config = config
        raise NotImplementedError(
            "LocalDiffusersGenerator is a skeleton. Use DummyImageGenerator for the first run."
        )

    def generate(self, prompt: str, output_path: str | Path) -> Path:
        """Generate an image with a local diffusion model.

        Args:
            prompt: Text prompt.
            output_path: Path where the generated image should be saved.

        Returns:
            Path to the saved image.
        """
        raise NotImplementedError


def build_image_generator(config: dict) -> DummyImageGenerator:
    """Build the image generator selected by the config.

    Args:
        config: Loaded project configuration.

    Returns:
        Image generator instance.
    """
    generator_config = config.get("image_generator", {})
    provider = generator_config.get("provider", "dummy")

    if provider != "dummy":
        print("Only the dummy image generator is runnable in this first version.")

    return DummyImageGenerator(
        width=int(generator_config.get("width", 512)),
        height=int(generator_config.get("height", 512)),
    )
