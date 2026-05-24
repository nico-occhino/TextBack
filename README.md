# TextBack

TextBack is an Advanced Deep Learning exam project for probing spurious visual
features in ImageNet classifiers.  The real optimization backend uses the
official TextGrad library: prompt variable, generated image, classifier
feedback, `TextLoss`, `loss.backward()`, and `TGD.step()`.

The default config is intentionally tiny: 1 target class, 1 optimization step,
and 1 inference image.  This avoids wasting GPU time and API calls while
checking that the full pipeline is connected.

## Environment

Create a local `.env` file from the example:

```bash
copy .env.example .env
```

Then edit `.env` and set:

```text
GEMINI_API_KEY=your_real_key_here
```

Do not commit `.env`.  It is ignored by git.

Check the environment:

```bash
python scripts/check_environment.py
```

This prints the Python version, torch/CUDA status, selected GPU, whether a
Gemini key is present, and whether `textgrad` and `diffusers` import.

## Tiny Run

Run one small optimization:

```bash
python scripts/run_optimization.py --config configs/default.yaml
```

Then generate one inference image from the final prompt:

```bash
python scripts/run_inference.py --config configs/default.yaml
```

Outputs are written under `results/`, including optimization logs, final
prompts, inference results, and generated images.

## Useful Helpers

List exact ImageNet labels:

```bash
python scripts/list_imagenet_classes.py --query bus
python scripts/list_imagenet_classes.py --query dog
```

## Main Files

```text
configs/default.yaml          experiment settings
src/classifier.py             dummy and torchvision ImageNet classifiers
src/image_generator.py        dummy and local Diffusers image generators
src/textgrad_optimizer.py     official TextGrad prompt optimizer
src/textual_backward.py       older custom prompt optimizer
src/pipeline.py               optimization and inference orchestration
scripts/run_optimization.py   optimization entry point
scripts/run_inference.py      inference entry point
scripts/check_environment.py  dependency and key checker
```
