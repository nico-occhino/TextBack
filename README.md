# TextBack

TextBack is an Advanced Deep Learning exam project for Project 17, Textual Backward. It optimizes text-to-image prompts so generated images activate a fixed robust ImageNet classifier.

The project does not train Stable Diffusion or RobustResNet50. Stable Diffusion and RobustBench are frozen. The optimized variable is the text prompt. TextGrad produces natural-language feedback, not numerical gradients.

## Pipeline

```text
LLM initial prompt
-> Stable Diffusion image generation
-> RobustBench RobustResNet50 classification
-> TextGrad textual refinement
-> positive descriptor memory
-> best-so-far prompt selection
-> inference AMR@1 / AMR@5
-> optional post-hoc Grad-CAM and occlusion
```

Prompts are name-free: they must avoid the exact target class name and close synonyms. The lexical guardrail checks exact forbidden terms, not every possible semantic cue.

## Code Organization

```text
configs/default.yaml              small development run
configs/final.yaml                submitted final run
src/config.py                     YAML loading and validation
src/classifier.py                 frozen RobustBench classifier
src/image_generator.py            frozen Stable Diffusion generator
src/textgrad_optimizer.py         OpenAI/TextGrad prompt refinement
src/descriptors.py                positive descriptor memory
src/pipeline.py                   optimization and inference orchestration
scripts/run_optimization.py       optimize prompts
scripts/run_inference.py          generate inference images and AMR metrics
scripts/imagenet_subset.py        prepare the real ImageNet subset
scripts/evaluate_real_subset.py   evaluate real ImageNet subset baseline
scripts/run_xai.py                compute post-hoc XAI artifacts
scripts/build_final_results.py    build final tables, PNG figures, and summary
scripts/dev/                      setup and debugging helpers
```

Small utility helpers are kept close to the code that uses them. The main scientific modules stay separate.

## Environment

Create `.env` from `.env.example` without committing real keys:

```text
HF_TOKEN=your_huggingface_token_here
OPENAI_API_KEY=your_openai_api_key_here
```

`OPENAI_API_KEY` is required for TextGrad/OpenAI calls. `HF_TOKEN` is optional but recommended for Hugging Face downloads and rate limits.

PowerShell example:

```powershell
$env:HF_TOKEN="hf_..."
$env:OPENAI_API_KEY="sk-..."
```

Install dependencies from `requirements.txt`. RobustBench may also need its model checkpoint downloaded on first use.

Useful development checks:

```bash
python scripts/dev/check_environment.py
python scripts/dev/test_llm_backend.py --config configs/final.yaml
python scripts/dev/list_imagenet_classes.py --query bus
```

## OpenAI Backend

The submitted code path is OpenAI-only:

```yaml
textgrad:
  backward_engine: "experimental/gpt-5.4-mini"
```

`src/textgrad_optimizer.py` converts this to the LiteLLM model name `openai/gpt-5.4-mini` and requires `OPENAI_API_KEY`.

Advanced users can adapt the project to Groq, Gemini, or another LiteLLM provider by changing `textgrad.backward_engine` in the YAML config and reintroducing the corresponding provider API-key check. That is outside the submitted OpenAI-only code path.

## Final Reproduction Commands

```powershell
python scripts/run_optimization.py --config configs/final.yaml
python scripts/run_inference.py --config configs/final.yaml
python scripts/run_xai.py --config configs/final.yaml --patch-size 56 --stride 28 --max-images-per-class 100
python scripts/build_final_results.py --config configs/final.yaml --output-dir results/final_report_assets
```

Existing saved optimization and inference results can be reused, so only the missing downstream steps need to be rerun. `run_xai.py` writes machine-readable XAI artifacts under `results/xai/`; `build_final_results.py` is the only script that creates final report tables and figures.

To prepare and evaluate the real-subset baseline:

```powershell
python scripts/imagenet_subset.py --output-dir data/imagenet_subset --n-per-class 50
python scripts/evaluate_real_subset.py --config configs/final.yaml
```

## Result Files

TextBack reads and writes result artifacts under `results/`. Existing local results are not required to be committed.

Important files:

```text
results/initial_prompts.json          cached LLM initial prompts
results/initial_prompt_metadata.json  prompt source and guardrail metadata
results/final_prompts.json            best prompts after optimization
results/best_prompt_metadata.json     best iteration and target confidence
results/descriptor_memory.json        positive descriptor memory
results/optimization_logs.csv         optimization trajectory
results/inference_results.csv         per-image inference predictions
results/inference_summary.json        AMR@1, AMR@5, and internal diagnostics
results/activation_rates.json         AMR@1 kept for compatibility
results/real_subset_summary.csv       real-image baseline summary
results/xai/occlusion_results.csv     per-image occlusion statistics
results/xai/occlusion_summary.csv     per-class occlusion summary
results/xai/xai_selected_examples.csv selected qualitative XAI examples
results/xai/selected/                 selected Grad-CAM and occlusion arrays
results/final_report_assets/          final CSV/LaTeX tables and PNG figures
```

Final report figures are PNG only. Historical files elsewhere under `results/` are left in place for reproducibility.

## Metrics

AMR@1 is the primary activation maximization metric: the fraction of generated images where `target_rank == 1`.

AMR@5 is the secondary metric: the fraction where `target_rank <= 5`.

The oral-exam story also uses the real-subset baseline, confusion distribution, optimization trajectory, guardrail rejections, best-prompt metadata, initial prompt metadata, and descriptor memory. Mean confidence and mean rank are saved as internal diagnostics but are not the main result.

## Interpretation

Grad-CAM localizes regions supporting the target-class logit. Occlusion sensitivity estimates whether masking local regions decreases target confidence. Occlusion is stronger than Grad-CAM because it is perturbation-based, but it is still not definitive causal proof because masking introduces distribution shift. These analyses support candidate classifier-salient cue discovery.

Positive descriptor memory preserves short cue phrases from high-activation iterations so later TextGrad updates are less likely to discard useful evidence. It is deterministic memory, not another LLM agent.

TextGrad can occasionally return malformed optimizer responses when the textual context is too long or the LLM does not follow the expected format. TextBack treats those cases as rejected updates and keeps the previous prompt.

## Limits

Stable Diffusion 1.5 has a short CLIP text window, so prompts are capped with `textgrad.max_prompt_words`.

The real ImageNet subset is only a classifier sanity-check baseline. It is not training data.
