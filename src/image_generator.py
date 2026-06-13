"""Image generation for TextBack using Hugging Face Diffusers."""

from pathlib import Path


class LocalDiffusersGenerator:
    """Local Stable Diffusion generator based on Hugging Face Diffusers."""

    def __init__(self, config: dict) -> None:
        """Load a Diffusers text-to-image pipeline."""
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
        """Generate one image from a prompt and save it."""
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
        from PIL import ImageStat

        grayscale = image.convert("L")
        mean_pixel = ImageStat.Stat(grayscale).mean[0]

        if mean_pixel < 2:
            print(
                "Generated image appears almost black. Consider force_float32=true, "
                "disable_safety_checker=true, or changing the prompt/seed."
            )


def build_image_generator(config: dict):
    """Create the configured image generator."""
    generator_config = config["image_generator"]
    provider = generator_config.get("provider", "diffusers")

    if provider == "diffusers":
        return LocalDiffusersGenerator(config)

    raise ValueError("Final TextBack workflow requires image_generator.provider='diffusers'.")
