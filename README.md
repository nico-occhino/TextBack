# TextBack

TextBack is an Advanced Deep Learning oral exam project for **Project 17:
Textual Backward**.  It uses TextGrad to optimize text-to-image prompts that
search for name-free visual cues capable of activating a robust ImageNet
classifier.

The goal is not to prove causality.  The goal is to discover candidate
textures, materials, shapes, colors, backgrounds, and co-occurring cues that may
act like shortcut features for a fixed classifier.

## Final Pipeline

```text
TextGrad prompt variable
  -> Stable Diffusion image generation
  -> RobustBench Salman2020Do_R50 classifier prediction
  -> TextGrad TextLoss from classifier feedback
  -> loss.backward()
  -> TextGrad TGD optimizer.step()
  -> updated name-free cue prompt
  -> inference activation maximization
```

Stable Diffusion and RobustBench are fixed black-box forward components.  Only
the prompt changes.  TextGrad provides the PyTorch-like pieces: `Variable`,
`TextLoss`, `backward()`, and `TGD.step()`.  The gradient is natural-language
feedback, not a numerical derivative.

## Prompt Methodology

The workflow is **name-free cue optimization**.  Image-generation prompts must
avoid exact target class names and close synonyms, but they may use visual
attributes, textures, materials, shapes, colors, context, and co-occurring cues.
Hard lexical guardrails reject TextGrad updates that leak target names.

Prompt files live in `prompts/`:

```text
prompts/initial_prompt_system.txt      methodology for initial seed prompts
prompts/refinement_prompt_system.txt   active TextGrad refinement instruction
prompts/textual_loss_system.txt        future-work note, not active
```

Initial prompts are deterministic class-specific seed prompts in
`src/pipeline.py`.  They are aligned with `initial_prompt_system.txt` for
reproducibility and easy oral-exam explanation.  `refinement_prompt_system.txt`
is loaded and used inside the TextGrad loss instruction.  `textual_loss_system`
documents a possible future separate critic but is not implemented.

Stable Diffusion 1.5 has a short CLIP text context window, so prompts are capped
by `textgrad.max_prompt_words`.

TextGrad updates are not guaranteed to improve activation monotonically.  During
optimization, TextBack therefore keeps the prompt with the highest observed
target confidence and uses that best-so-far prompt for final inference.  This is
analogous to saving the best checkpoint during model training.

## Configs

`configs/default.yaml` is for development:

```text
n_optimization_steps: 3
n_inference_images: 20
```

`configs/final.yaml` is the final/spec-compliant run:

```text
n_optimization_steps: 5
n_inference_images: 100
max_prompt_words: 45
```

Both configs use:

```text
TextGrad backend: experimental:groq/openai/gpt-oss-20b
Image generator: runwayml/stable-diffusion-v1-5
Classifier: RobustBench Salman2020Do_R50
```

## RobustBench Classifier

The visual classifier is RobustBench `Salman2020Do_R50`, a robust ImageNet
ResNet-50 model.  Torchvision ResNet50 is no longer the final classifier.

Install RobustBench with:

```bash
pip install "setuptools<81"
pip install "gdown>=5.2.0"
pip install git+https://github.com/RobustBench/robustbench.git
```

The first model load downloads the checkpoint under `models/imagenet/Linf/`.
The `models/` folder is ignored by Git.

## API Key

Create `.env` manually from `.env.example`:

```text
GROQ_API_KEY=your_real_key_here
```

Never commit real API keys.  ChatGPT Plus does not include API usage; API keys
and billing are separate.

## Environment Check

```bash
python scripts/check_environment.py
```

This checks Python, torch/CUDA, Groq key presence, RobustBench imports,
TextGrad, and Diffusers.  It does not load the full RobustBench checkpoint.

## ImageNet Subset Baseline

Torchvision can download model weights, but it does **not** automatically
download ImageNet / ILSVRC2012.  The real 5x50 subset is a classifier sanity
check baseline, not training data.

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

Useful commands:

```bash
python scripts/list_imagenet_classes.py --query bus
python scripts/evaluate_real_subset.py --config configs/default.yaml
```

## Running TextBack

Development run:

```bash
python scripts/run_optimization.py --config configs/default.yaml
python scripts/run_inference.py --config configs/default.yaml
```

Final run:

```bash
python scripts/run_optimization.py --config configs/final.yaml
python scripts/run_inference.py --config configs/final.yaml
```

Inspect saved results:

```bash
python scripts/inspect_results.py --config configs/default.yaml
```

The main metric is activation maximization rate.  Inference summary metrics
such as top-5 activation and mean target confidence are diagnostics.

## Main Files

```text
configs/default.yaml             lightweight development settings
configs/final.yaml               final/spec-compliant settings
src/classifier.py                RobustBench Salman2020Do_R50 classifier
src/image_generator.py           Diffusers image generator
src/textgrad_optimizer.py        TextGrad name-free cue optimization
src/pipeline.py                  optimization and inference orchestration
scripts/check_environment.py     dependency and key checker
scripts/list_imagenet_classes.py ImageNet label lookup
scripts/evaluate_real_subset.py  real ImageNet subset baseline evaluation
scripts/run_optimization.py      TextBack optimization entry point
scripts/run_inference.py         TextBack inference entry point
scripts/inspect_results.py       compact result summary
```
