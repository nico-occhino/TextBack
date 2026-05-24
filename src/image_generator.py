"""Image generation utilities for TextBack.

The real experiment uses a local Diffusers pipeline.  The dummy generator stays
available for testing the project plumbing without a GPU or model download.
"""

from pathlib import Path
import hashlib
from textwrap import wrap
import struct
import zlib


class DummyImageGenerator:
    """Create simple placeholder images from prompts.

    This generator is only for testing the pipeline plumbing.  Its images are
    not meaningful samples and should not be used for scientific evaluation.
    """

    def __init__(self, width: int = 512, height: int = 512, draw_text: bool = False) -> None:
        """Store output image size.

        Args:
            width: Image width in pixels.
            height: Image height in pixels.
            draw_text: If True, write the prompt on the dummy image.
        """
        self.width = width
        self.height = height
        self.draw_text = draw_text

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
            if self.draw_text:
                self._generate_text_image(prompt, output_path)
            else:
                self._generate_shape_image(prompt, output_path)
        except ModuleNotFoundError:
            # Fallback for bare Python environments.  It cannot draw text, but
            # it still produces a valid RGB PNG with a central geometric object.
            self._generate_plain_shape_png(prompt, output_path)

        return output_path

    def _generate_text_image(self, prompt: str, output_path: Path) -> None:
        """Use PIL to create a placeholder image with prompt text.

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

    def _generate_shape_image(self, prompt: str, output_path: Path) -> None:
        """Use PIL to create a text-free geometric dummy image.

        Args:
            prompt: Prompt used to choose deterministic colors.
            output_path: Image path to save.
        """
        from PIL import Image, ImageDraw

        background_color = self._color_from_prompt(prompt)
        object_color = self._contrast_color(background_color)
        image = Image.new("RGB", (self.width, self.height), background_color)
        draw = ImageDraw.Draw(image)

        # A plain central object is less likely than text to trigger labels such
        # as "web site" when experimenting with a real classifier.
        margin_x = self.width // 4
        margin_y = self.height // 4
        draw.ellipse(
            (margin_x, margin_y, self.width - margin_x, self.height - margin_y),
            fill=object_color,
            outline="white",
            width=4,
        )
        draw.rectangle(
            (self.width // 2 - 35, self.height // 2 - 35, self.width // 2 + 35, self.height // 2 + 35),
            fill=background_color,
            outline="white",
            width=3,
        )

        image.save(output_path)

    def _generate_plain_shape_png(self, prompt: str, output_path: Path) -> None:
        """Write a simple geometric PNG without external dependencies.

        Args:
            prompt: Prompt used to choose deterministic colors.
            output_path: Image path to save.
        """
        background_color = self._color_from_prompt(prompt)
        object_color = self._contrast_color(background_color)
        raw_rows = []

        center_x = self.width / 2
        center_y = self.height / 2
        radius = min(self.width, self.height) / 4

        for y in range(self.height):
            row_pixels = bytearray()
            for x in range(self.width):
                distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
                color = object_color if distance < radius else background_color
                row_pixels.extend(color)
            raw_rows.append(b"\x00" + bytes(row_pixels))

        raw_pixels = b"".join(raw_rows)

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
        digest = hashlib.md5(prompt.encode("utf-8")).digest()
        return (70 + digest[0] % 120, 70 + digest[1] % 120, 70 + digest[2] % 120)

    def _contrast_color(self, color: tuple[int, int, int]) -> tuple[int, int, int]:
        """Create a visible foreground color for the central object.

        Args:
            color: Background RGB color.

        Returns:
            Contrasting RGB color.
        """
        red, green, blue = color
        return (255 - red, 255 - green, 255 - blue)


class LocalDiffusersGenerator:
    """Local Stable Diffusion generator based on Hugging Face Diffusers."""

    def __init__(self, config: dict) -> None:
        """Load a Diffusers text-to-image pipeline.

        Args:
            config: Loaded project configuration.
        """
        import torch
        from diffusers import DiffusionPipeline

        self.torch = torch
        self.config = config
        self.generator_config = config.get("image_generator", {})

        requested_device = config.get("project", {}).get("device", "cpu")
        cuda_available = torch.cuda.is_available()
        self.device = "cuda" if requested_device == "cuda" and cuda_available else "cpu"

        force_float32 = bool(self.generator_config.get("force_float32", False))
        use_float16 = bool(self.generator_config.get("use_float16", True))
        if force_float32:
            dtype = torch.float32
        elif use_float16 and self.device == "cuda":
            dtype = torch.float16
        else:
            dtype = torch.float32

        model_name = self.generator_config["model_name"]
        disable_safety_checker = bool(self.generator_config.get("disable_safety_checker", False))

        load_kwargs = {"torch_dtype": dtype}
        if disable_safety_checker:
            # Only disable this for controlled academic experiments.  Do not
            # use this option for public image generation systems.
            load_kwargs["safety_checker"] = None

        try:
            self.pipe = DiffusionPipeline.from_pretrained(model_name, **load_kwargs)
        except TypeError:
            # Some pipelines do not accept safety_checker at load time.
            load_kwargs.pop("safety_checker", None)
            self.pipe = DiffusionPipeline.from_pretrained(model_name, **load_kwargs)

        if bool(self.generator_config.get("enable_attention_slicing", False)):
            self.pipe.enable_attention_slicing()

        # Only disable this for controlled academic experiments.  Do not use
        # this option for public image generation systems.
        if disable_safety_checker and hasattr(self.pipe, "safety_checker"):
            self.pipe.safety_checker = None

        # CPU offload can save VRAM, but it requires accelerate and a compatible
        # pipeline.  If unavailable, we fall back to the normal .to(device).
        use_cpu_offload = bool(self.generator_config.get("enable_cpu_offload", False))
        if use_cpu_offload and hasattr(self.pipe, "enable_model_cpu_offload"):
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(self.device)

    def generate(self, prompt: str, output_path: str | Path) -> Path:
        """Generate an image with the local diffusion model.

        Args:
            prompt: Text prompt.
            output_path: Path where the generated image should be saved.

        Returns:
            Path to the saved image.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        image = self.pipe(
            prompt,
            height=int(self.generator_config.get("height", 384)),
            width=int(self.generator_config.get("width", 384)),
            num_inference_steps=int(self.generator_config.get("num_inference_steps", 10)),
            guidance_scale=float(self.generator_config.get("guidance_scale", 7.5)),
        ).images[0]

        self._warn_if_almost_black(image)
        image.save(output_path)
        return output_path

    def _warn_if_almost_black(self, image) -> None:
        """Warn when Diffusers returns an almost black image.

        Args:
            image: PIL image returned by the diffusion pipeline.
        """
        grayscale = image.convert("L")
        pixels = list(grayscale.getdata())
        mean_pixel = sum(pixels) / len(pixels)

        if mean_pixel < 2:
            print(
                "Generated image appears almost black. Consider force_float32=true, "
                "disable_safety_checker=true, or changing the prompt/seed."
            )


def build_image_generator(config: dict):
    """Build the image generator selected by the config.

    Args:
        config: Loaded project configuration.

    Returns:
        Image generator instance.
    """
    generator_config = config.get("image_generator", {})
    provider = generator_config.get("provider", "dummy")

    if provider == "diffusers":
        return LocalDiffusersGenerator(config)

    if provider != "dummy":
        print("Unknown image generator provider. Falling back to dummy generator.")

    return DummyImageGenerator(
        width=int(generator_config.get("width", 512)),
        height=int(generator_config.get("height", 512)),
        draw_text=bool(generator_config.get("draw_text_on_dummy_image", False)),
    )
