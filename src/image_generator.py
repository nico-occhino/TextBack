"""Image generators for TextBack.

The real experiments use Diffusers.  The dummy generator is kept only for quick
development checks when we do not want to load a diffusion model.
"""

from pathlib import Path
import hashlib


class DummyImageGenerator:
    """Development-only generator that writes a simple colored shape."""

    def __init__(self, width: int = 384, height: int = 384) -> None:
        """Store image size.

        Args:
            width: Output width in pixels.
            height: Output height in pixels.
        """
        self.width = width
        self.height = height

    def generate(self, prompt: str, output_path: str | Path, seed: int | None = None) -> Path:
        """Create a simple placeholder image without prompt text.

        Args:
            prompt: Prompt used only to choose deterministic colors.
            output_path: Path where the image should be saved.
            seed: Optional seed, accepted for interface compatibility.

        Returns:
            Path to the saved image.
        """
        from PIL import Image, ImageDraw

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        background_color = self._color_from_prompt(prompt)
        object_color = tuple(255 - value for value in background_color)
        image = Image.new("RGB", (self.width, self.height), background_color)
        draw = ImageDraw.Draw(image)

        margin_x = self.width // 4
        margin_y = self.height // 4
        draw.ellipse(
            (margin_x, margin_y, self.width - margin_x, self.height - margin_y),
            fill=object_color,
            outline="white",
            width=4,
        )

        image.save(output_path)
        return output_path

    def _color_from_prompt(self, prompt: str) -> tuple[int, int, int]:
        """Create a deterministic RGB color from prompt text."""
        digest = hashlib.md5(prompt.encode("utf-8")).digest()
        return (70 + digest[0] % 120, 70 + digest[1] % 120, 70 + digest[2] % 120)


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
        self.generator_config = config["image_generator"]

        requested_device = config["project"].get("device", "cpu")
        self.device = "cuda" if requested_device == "cuda" and torch.cuda.is_available() else "cpu"
        dtype = self._select_dtype(torch)
        model_name = self.generator_config["model_name"]
        disable_safety_checker = bool(self.generator_config.get("disable_safety_checker", False))

        load_kwargs = {"torch_dtype": dtype}
        if disable_safety_checker:
            # Only for controlled academic experiments. Do not disable safety
            # checks in public image generation systems.
            load_kwargs["safety_checker"] = None

        try:
            self.pipe = DiffusionPipeline.from_pretrained(model_name, **load_kwargs)
        except TypeError:
            load_kwargs.pop("safety_checker", None)
            self.pipe = DiffusionPipeline.from_pretrained(model_name, **load_kwargs)

        if bool(self.generator_config.get("enable_attention_slicing", False)):
            self.pipe.enable_attention_slicing()

        if disable_safety_checker and hasattr(self.pipe, "safety_checker"):
            self.pipe.safety_checker = None

        use_cpu_offload = bool(self.generator_config.get("enable_cpu_offload", False))
        if use_cpu_offload and hasattr(self.pipe, "enable_model_cpu_offload"):
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(self.device)

    def generate(self, prompt: str, output_path: str | Path, seed: int | None = None) -> Path:
        """Generate one image from a prompt and save it.

        Args:
            prompt: Text-to-image prompt.
            output_path: Path where the generated image should be saved.
            seed: Optional deterministic seed for Diffusers generation.

        Returns:
            Path to the saved image.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        generator = None
        if seed is not None:
            generator = self.torch.Generator(device=self.device).manual_seed(seed)

        image = self.pipe(
            prompt,
            height=int(self.generator_config.get("height", 384)),
            width=int(self.generator_config.get("width", 384)),
            num_inference_steps=int(self.generator_config.get("num_inference_steps", 10)),
            guidance_scale=float(self.generator_config.get("guidance_scale", 7.5)),
            generator=generator,
        ).images[0]

        self._warn_if_almost_black(image)
        image.save(output_path)
        return output_path

    def _select_dtype(self, torch):
        """Choose the dtype used to load the diffusion model."""
        if bool(self.generator_config.get("force_float32", False)):
            return torch.float32
        if bool(self.generator_config.get("use_float16", True)) and self.device == "cuda":
            return torch.float16
        return torch.float32

    def _warn_if_almost_black(self, image) -> None:
        """Warn when a generated image looks almost entirely black."""
        grayscale = image.convert("L")
        pixels = list(grayscale.getdata())
        mean_pixel = sum(pixels) / len(pixels)

        if mean_pixel < 2:
            print(
                "Generated image appears almost black. Consider force_float32=true, "
                "disable_safety_checker=true, or changing the prompt/seed."
            )


def build_image_generator(config: dict):
    """Create the configured image generator.

    Args:
        config: Loaded project configuration.

    Returns:
        LocalDiffusersGenerator or development-only DummyImageGenerator.
    """
    generator_config = config["image_generator"]
    provider = generator_config.get("provider", "diffusers")

    if provider == "diffusers":
        return LocalDiffusersGenerator(config)
    if provider == "dummy":
        return DummyImageGenerator(
            width=int(generator_config.get("width", 384)),
            height=int(generator_config.get("height", 384)),
        )

    raise ValueError(f"Unsupported image generator provider: {provider}")
