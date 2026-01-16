# LLaVA-Med (SLAKE fine-tuning & evaluation toolkit)

This repository provides training, evaluation, and data-prep utilities for fine-tuning and analyzing LLaVA-Med models on the SLAKE medical VQA dataset. It includes end-to-end scripts for supervised fine-tuning, OPEN vs. CLOSED answer routing, optional segmentation-mask conditioning, and TensorBoard diagnostics on SLAKE splits. The repo also vendors the upstream LLaVA-Med codebase for model loading and inference utilities.

## What’s in this repo

### Core workflows

- **Training (`train_llava_med.py`)**
  - Loads SLAKE from Hugging Face or local JSON files, filters to English samples, and optionally injects KG triples into prompts.
  - Builds mixed OPEN/CLOSED prompts, supports prompt templates for short answers, and handles yes/no normalization for closed questions.
  - Provides optional mask handling (crop/masked) using SLAKE segmentation maps or legacy mask files.
  - Supports LoRA fine-tuning, optional vision tower freezing, and configurable multimodal projector tuning.
  - Logs rich TensorBoard diagnostics through a dedicated VQA callback (dataset stats, text previews, image grids, loss histograms).

- **Evaluation (`eval_llava_med.py`)**
  - Runs SLAKE evaluation with separate OPEN vs. CLOSED scoring pipelines.
  - Supports CLOSED candidate routing (yes/no, explicit option parsing, vocab fallback) and answer normalization.
  - Emits evaluation summaries, cached vocab files, and optional diagnosis artifacts (prompt dumps, mask debug images, error samples).

- **Data conversion (`slake_to_llava_sft.py`)**
  - Converts SLAKE splits to LLaVA-style supervised fine-tuning JSON with conversations and image paths.

- **Sanity check (`test.py`)**
  - Verifies model loading path, trainable parameter counts, image token placement, and runs a minimal generate pass.

### Repository layout

- `train_llava_med.py` – training entrypoint for SLAKE fine-tuning.
- `eval_llava_med.py` – evaluation script with OPEN/CLOSED routing.
- `slake_to_llava_sft.py` – SLAKE-to-LLaVA SFT conversion.
- `data/` – SLAKE loading utilities and the LLaVA SFT collator.
- `models/` – LLaVA(-Med) wrappers for loading official LLaVA models and generating VQA outputs.
- `utils/` – prompt builders, mask preprocessing, closed-candidate routing, LoRA utilities, and normalization helpers.
- `callbacks/` – TensorBoard callback for VQA metrics.
- `tests/` – unit tests for prompt formatting, mask handling, routing, and scoring.
- `LLaVA-Med/` and `src/llava-med/` – vendored upstream LLaVA-Med codebase used for official model loading.

## Setup

1. **Python environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install upstream LLaVA-Med (local editable)**

   The training/evaluation pipeline relies on `llava` modules from the upstream LLaVA-Med codebase. If it is not already on your Python path, install the vendored copy:

   ```bash
   pip install -e LLaVA-Med
   ```

## Data expectations (SLAKE)

The scripts assume either:

- **Hugging Face mode**: `datasets.load_dataset("BoKelvin/SLAKE")`, or
- **Local mode**: a `SLAKE/` folder with `train.json`, `validation.json`, `test.json`, and `img/` images.

Optional mask data is discovered by searching typical directories like `mask/`, `masks/`, `seg/`, or alongside the images, plus an optional `mask.txt` for class-id label mapping.

## Training

Minimal training run (local SLAKE data):

```bash
python train_llava_med.py \
  --data_source local \
  --slake_root SLAKE \
  --output_dir work_slake_llava_med_train
```

Useful flags:

- `--mask_mode {none,crop,masked}` to condition on segmentation masks.
- `--use_lora` / `--lora_target_modules` to LoRA-finetune selective layers.
- `--freeze_vision_tower` and `--tune_mm_projector` for multimodal tuning control.
- `--tb` and related `--tb_*` options for TensorBoard logging.

## Evaluation

Evaluate a checkpoint or HF model on SLAKE:

```bash
python eval_llava_med.py \
  --model_id_or_ckpt microsoft/llava-med-v1.5-mistral-7b \
  --data_source local \
  --slake_root SLAKE \
  --output_dir work_slake_llava_med_eval
```

Useful flags:

- `--closed_candidate_mode` to choose CLOSED routing logic.
- `--closed_vocab_path` to reuse a saved CLOSED vocab.
- `--mode diagnose` to dump diagnostic artifacts.

## SLAKE → LLaVA SFT conversion

```bash
python slake_to_llava_sft.py \
  --data_root SLAKE \
  --output_dir work_slake_llava_med
```

## Sanity check

```bash
python test.py --model_id microsoft/llava-med-v1.5-mistral-7b
```

## Tests

```bash
pytest
```

## Notes & references

- The upstream LLaVA-Med repository and documentation live in `LLaVA-Med/` and `src/llava-med/`.
- SLAKE is a medical VQA dataset; ensure you comply with its license and usage restrictions.
