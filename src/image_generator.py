from pathlib import Path


class LocalDiffusersGenerator:

    def __init__(self, config: dict) -> None:
        import torch
        from diffusers import DiffusionPipeline

        self.torch = torch
        self.generator_config = config["image_generator"]

        requested_device = config["project"].get("device", "cpu")
        self.device = "cuda" if requested_device == "cuda" and torch.cuda.is_available() else "cpu"

        self.pipe = DiffusionPipeline.from_pretrained(
            self.generator_config["model_name"],
            torch_dtype=torch.float32,
            safety_checker=None,
        )

        self.pipe.enable_attention_slicing()

        if hasattr(self.pipe, "enable_model_cpu_offload"):
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(self.device)

        if hasattr(self.pipe, "safety_checker"):
            self.pipe.safety_checker = None

    def generate(self, prompt: str, output_path: str | Path, seed: int) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        generator = self.torch.Generator(device=self.device).manual_seed(seed)

        image = self.pipe(
            prompt,
            height=int(self.generator_config.get("height", 384)),
            width=int(self.generator_config.get("width", 384)),
            num_inference_steps=int(self.generator_config.get("num_inference_steps", 30)),
            guidance_scale=float(self.generator_config.get("guidance_scale", 7.5)),
            generator=generator,
        ).images[0]

        self._warn_if_almost_black(image)
        image.save(output_path)
        return output_path

    def _warn_if_almost_black(self, image) -> None:
        from PIL import ImageStat

        grayscale = image.convert("L")
        mean_pixel = ImageStat.Stat(grayscale).mean[0]

        if mean_pixel < 2:
            print("Generated image appears almost black. Check dtype, prompt, or seed.")


def build_image_generator(config: dict) -> LocalDiffusersGenerator:
    return LocalDiffusersGenerator(config)