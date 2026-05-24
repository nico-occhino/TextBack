# TextBack

TextBack is an Advanced Deep Learning oral exam project for **Project 17:
Textual Backward**.  The goal is to find shortcut cues and spurious visual
features that activate a robust pretrained ImageNet classifier.

This final version is **shortcut-only**.  The image-generation prompt must not
directly name the target class.  The target class is used only inside the
TextGrad textual loss and the classifier feedback.

## Final Pipeline

```text
TextGrad prompt variable
  -> local Diffusers image generation
  -> RobustBench Salman2020Do_R50 ImageNet classifier
  -> TextGrad TextLoss from classifier feedback
  -> loss.backward()
  -> TextGrad TGD optimizer.step()
  -> updated shortcut prompt
```

The final visual classifier is `Salman2020Do_R50` from RobustBench.  This is a
robust ImageNet ResNet-50 model.  Torchvision ResNet50 is no longer the final
classifier for this project.

## Methodological Guardrails

TextBack is shortcut-only, so target leakage is a real risk.  The optimizer uses
hard lexical guardrails: if TextGrad proposes a prompt containing target labels
or direct object parts, the update is rejected and the previous prompt is kept.

Inference uses different deterministic seeds for each generated sample.  This
keeps samples reproducible while avoiding identical generated images.

## RobustBench Install

Install RobustBench with:

```bash
pip install "setuptools<81"
pip install "gdown>=5.2.0"
pip install git+https://github.com/RobustBench/robustbench.git
```

If RobustBench installation or model loading fails, the project environment is
not ready for the final classifier.

## API Key

Create `.env` manually from `.env.example` and add your Groq key:

```text
GROQ_API_KEY=your_real_key_here
```

Never commit real API keys.  ChatGPT Plus does not include API usage; API keys
and billing are separate.

The working TextGrad backend in the default config is:

```text
experimental:groq/openai/gpt-oss-20b
```

## Environment Check

```bash
python scripts/check_environment.py
```

This checks Python, torch/CUDA, Groq key presence, TextGrad, Diffusers, and
RobustBench.

## ImageNet Labels

Target classes and subset folder names must match Torchvision ImageNet labels
exactly:

```bash
python scripts/list_imagenet_classes.py --query bus
```

## Real ImageNet Subset

Torchvision can download model weights, but it does **not** automatically
download ImageNet / ILSVRC2012.  The real 5x50 subset is a baseline for
evaluation only, not training.

Expected structure:

```text
data/imagenet_subset/<exact ImageNet label>/*.jpg
data/imagenet_subset/<exact ImageNet label>/*.png
```

Example:

```text
data/imagenet_subset/
  tabby/
    001.png
  sports car/
    001.png
```

Evaluate the real subset:

```bash
python scripts/evaluate_real_subset.py --config configs/default.yaml
```

## TextBack Runs

Run prompt optimization:

```bash
python scripts/run_optimization.py --config configs/default.yaml
```

Run inference from the final prompts:

```bash
python scripts/run_inference.py --config configs/default.yaml
```

Outputs are written under `results/`.

## Main Files

```text
configs/default.yaml             experiment settings
src/classifier.py                RobustBench Salman2020Do_R50 classifier
src/image_generator.py           Diffusers generator
src/textgrad_optimizer.py        TextGrad shortcut prompt optimization
src/pipeline.py                  optimization and inference orchestration
scripts/check_environment.py     dependency and key checker
scripts/list_imagenet_classes.py ImageNet label lookup
scripts/evaluate_real_subset.py  real ImageNet subset baseline evaluation
scripts/run_optimization.py      TextBack optimization entry point
scripts/run_inference.py         TextBack inference entry point
```
