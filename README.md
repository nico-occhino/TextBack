# TextBack

TextBack is an Advanced Deep Learning exam project for probing spurious visual
features in ImageNet classifiers.  The final pipeline uses the official
TextGrad library, Groq/LiteLLM as the recommended TextGrad backward engine,
local Diffusers image generation, and Torchvision ResNet50.

The optimization loop is:

```text
TextGrad prompt variable
  -> Diffusers generated image
  -> ResNet50 ImageNet prediction
  -> TextGrad TextLoss from classifier feedback
  -> loss.backward()
  -> TextGrad TGD step
  -> updated prompt
```

The default config keeps the run small enough for iterative testing before
launching larger experiments.

## Environment

Create a local `.env` file from the example:

```bash
copy .env.example .env
```

Then edit `.env`:

```text
GROQ_API_KEY=your_real_groq_key_here
```

The recommended TextGrad backend is:

```text
experimental:groq/qwen/qwen3-32b
```

A cheaper alternative to try is:

```text
experimental:groq/openai/gpt-oss-20b
```

Gemini can still be used by changing `textgrad.backward_engine` in
`configs/default.yaml` and setting `GEMINI_API_KEY` or `GOOGLE_API_KEY`.
ChatGPT Plus does not include API usage; API keys and billing are separate.

Check the environment:

```bash
python scripts/check_environment.py
```

## ImageNet Labels

Target classes must match Torchvision ImageNet labels exactly.  Search labels
with:

```bash
python scripts/list_imagenet_classes.py --query bus
```

## Manual ImageNet Subset

Torchvision downloads pretrained model weights, but it does not download the
ImageNet / ILSVRC2012 dataset.  ImageNet must be obtained manually, or a small
subset must be created by hand.

For the project evaluation, use a folder like:

```text
data/imagenet_subset/
  school bus/
    img001.JPEG
    img002.JPEG
  golden retriever/
    img001.JPEG
```

The folder name is treated as the exact target label:

```text
data/imagenet_subset/<exact ImageNet label>/*.jpg
```

Evaluate the real subset:

```bash
python scripts/evaluate_real_subset.py --config configs/default.yaml
```

This writes `results/real_subset_predictions.csv` and
`results/real_subset_summary.csv`.

## TextBack Runs

Run one tiny optimization:

```bash
python scripts/run_optimization.py --config configs/default.yaml
```

Run inference from the final prompt:

```bash
python scripts/run_inference.py --config configs/default.yaml
```

Outputs are written under `results/`.

## Main Files

```text
configs/default.yaml             experiment settings
src/config.py                    YAML loading and validation
src/classifier.py                Torchvision ResNet50 ImageNet classifier
src/image_generator.py           Diffusers generator and dev dummy generator
src/textgrad_optimizer.py        TextGrad prompt optimization
src/pipeline.py                  optimization and inference orchestration
scripts/check_environment.py     dependency and key checker
scripts/list_imagenet_classes.py ImageNet label lookup
scripts/evaluate_real_subset.py  real ImageNet subset evaluation
scripts/run_optimization.py      TextBack optimization entry point
scripts/run_inference.py         TextBack inference entry point
```
