"""Train LLaVA-Med on SLAKE."""
from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from tqdm.auto import tqdm
from transformers import Trainer, TrainerCallback, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from data.collate_llava_sft import LlavaSFTCollator
from data.slake_dataset import SlakeDataset, load_slake_records, resolve_image_path
from models.llava_med_trainable import LlavaMedTrainable
from utils.closed_router import (
    build_closed_candidates,
    build_closed_vocab_from_train,
    canonicalize_yesno,
    find_option_target,
    sample_negatives,
    save_closed_vocab,
)
from utils.lora_utils import parse_lora_target_modules
from utils.mask_preprocess import (
    apply_mask,
    load_mask,
    load_segmentation_map,
    resolve_segmentation_mask_path,
)
from utils.probe_set import select_probe
from utils.tensorboard_utils import SafeSummaryWriter
from callbacks.vqa_tensorboard_callback import VQATensorBoardCallback
from utils.prompting import build_prompt, build_triples_context, build_user_text_closed

logger = logging.getLogger(__name__)


def _resize_mask_to_image(mask: np.ndarray, image: Image.Image) -> np.ndarray:
    """Resize a mask array to match the image size.

    Args:
        mask: Binary or integer mask array with shape (H, W).
        image: Target PIL image.

    Returns:
        Resized mask array aligned to the image.
    """
    if mask.shape[:2] == (image.size[1], image.size[0]):
        return mask
    mask_img = Image.fromarray(mask.astype("uint8") * 255)
    resized = mask_img.resize(image.size, resample=Image.NEAREST)
    return np.array(resized) > 0


def _mask_vis_from_seg_map(seg_map: Optional[np.ndarray], mask: np.ndarray, image: Image.Image) -> Image.Image:
    """Create a visualization image for segmentation or binary masks.

    Args:
        seg_map: Optional segmentation map with class ids.
        mask: Binary mask array.
        image: Original image for sizing.

    Returns:
        Grayscale visualization image for debugging masks.
    """
    if seg_map is None:
        resized_mask = _resize_mask_to_image(mask, image)
        vis = resized_mask.astype("uint8") * 255
    else:
        seg_map = _resize_mask_to_image(seg_map, image).astype("uint8") * 255 if seg_map.dtype == bool else seg_map
        seg_map = np.array(Image.fromarray(seg_map.astype("uint8")).resize(image.size, resample=Image.NEAREST))
        max_val = float(seg_map.max()) if seg_map.size else 0.0
        if max_val > 0:
            vis = (seg_map.astype("float32") / max_val * 255).astype("uint8")
        else:
            vis = seg_map.astype("uint8")
    return Image.fromarray(vis)


def _build_overlay(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """Overlay a red mask on top of an image for visualization.

    Args:
        image: Input PIL image.
        mask: Binary mask array.

    Returns:
        PIL image with a red overlay on masked regions.
    """
    resized_mask = _resize_mask_to_image(mask, image)
    overlay = np.array(image.convert("RGB")).astype("float32")
    color = np.array([255.0, 0.0, 0.0], dtype=np.float32)
    overlay[resized_mask] = overlay[resized_mask] * 0.6 + color * 0.4
    return Image.fromarray(overlay.astype("uint8"))


def _compute_mask_hit_rate(
    records: List[Dict],
    slake_root: Path,
    mask_threshold: int,
    mask_union_mode: str,
    sample_limit: int,
) -> float:
    """Compute the fraction of samples with available masks.

    Args:
        records: Dataset records to probe.
        slake_root: SLAKE dataset root directory.
        mask_threshold: Threshold for binary masks.
        mask_union_mode: Mask union strategy.
        sample_limit: Maximum number of records to inspect.

    Returns:
        Fraction of inspected samples that have a resolved mask.
    """
    if not records:
        return 0.0
    limit = min(len(records), sample_limit)
    found = 0
    for idx in range(limit):
        if load_mask(records[idx], slake_root, mask_threshold, mask_union_mode) is not None:
            found += 1
    return found / max(1, limit)


def _save_mask_debug_images(
    records: List[Dict],
    slake_root: Path,
    mask_mode: str,
    mask_threshold: int,
    mask_union_mode: str,
    mask_pad_ratio: float,
    output_dir: Path,
    max_items: int,
) -> int:
    """Save mask debug visualizations to disk.

    Args:
        records: Dataset records to visualize.
        slake_root: SLAKE dataset root directory.
        mask_mode: Masking mode for application.
        mask_threshold: Threshold for binary masks.
        mask_union_mode: Mask union strategy.
        mask_pad_ratio: Padding ratio for crop masks.
        output_dir: Output directory for debug images.
        max_items: Maximum number of records to render.

    Returns:
        Number of saved visualization sets.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for idx, record in enumerate(records):
        if saved >= max_items:
            break
        image_path = resolve_image_path(slake_root, str(record.get("img_name", "")))
        try:
            with Image.open(image_path) as img:
                image = img.convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError):
            continue
        mask = load_mask(record, slake_root, mask_threshold, mask_union_mode)
        if mask is None:
            continue
        seg_path = resolve_segmentation_mask_path(str(slake_root), str(record.get("img_name", "")))
        seg_map = load_segmentation_map(seg_path) if seg_path is not None else None
        masked = apply_mask(image, mask, mask_mode, mask_pad_ratio)
        mask_vis = _mask_vis_from_seg_map(seg_map, mask, image)
        overlay = _build_overlay(image, mask)
        qid = record.get("qid", idx)
        image.save(output_dir / f"{qid}_orig.png")
        mask_vis.save(output_dir / f"{qid}_mask_vis.png")
        masked.save(output_dir / f"{qid}_result.png")
        overlay.save(output_dir / f"{qid}_overlay.png")
        saved += 1
    return saved


def _select_diff_indices(total: int, count: int = 4) -> List[int]:
    """Select representative indices for mask pixel diff checks.

    Args:
        total: Total number of records.
        count: Maximum number of indices to return.

    Returns:
        List of unique indices spread across the dataset.
    """
    if total <= 0:
        return []
    candidates = [0, total // 3, total // 2, total - 1]
    unique = []
    for idx in candidates:
        if idx not in unique and 0 <= idx < total:
            unique.append(idx)
    return unique[:count]


def _run_mask_pixel_diff_check(
    model: LlavaMedTrainable,
    masked_dataset: SlakeDataset,
    unmasked_dataset: SlakeDataset,
    indices: List[int],
) -> float:
    """Compute the mean absolute difference between masked/unmasked pixels.

    Args:
        model: LLaVA-Med model wrapper.
        masked_dataset: Dataset with masks applied.
        unmasked_dataset: Dataset without masks.
        indices: Indices to compare for pixel differences.

    Returns:
        Mean absolute difference between pixel tensors.
    """
    if not indices:
        return 0.0
    masked_images = [masked_dataset[idx]["image"] for idx in indices]
    unmasked_images = [unmasked_dataset[idx]["image"] for idx in indices]
    with torch.no_grad():
        masked_pixels = model.processor(images=masked_images, return_tensors="pt")["pixel_values"]
        unmasked_pixels = model.processor(images=unmasked_images, return_tensors="pt")["pixel_values"]
        diff = torch.mean(torch.abs(masked_pixels - unmasked_pixels)).item()
    return diff


class VQATrainer(Trainer):
    """Trainer with optional CLOSED multiple-choice loss."""

    def __init__(
        self,
        *args,
        closed_vocab: Optional[Sequence[str]] = None,
        closed_train_objective: str = "hybrid",
        closed_mc_weight: float = 0.1,
        closed_mc_max_options: int = 6,
        closed_mc_vocab_k: int = 20,
        closed_mc_neg_seed: int = 0,
        closed_mc_routes: Sequence[str] = ("yesno", "options"),
        closed_use_vocab_fallback: bool = True,
        closed_yesno_variants: bool = True,
        closed_style: str = "minimal",
        triples_mode: str = "real_only",
        **kwargs,
    ) -> None:
        """Initialize the VQA trainer.

        Args:
            *args: Positional args forwarded to Trainer.
            closed_vocab: Vocabulary for CLOSED candidate sampling.
            closed_train_objective: ``sft``, ``mc``, or ``hybrid``.
            closed_mc_weight: Weight for the MC loss term.
            closed_mc_max_options: Maximum options for explicit-option routes.
            closed_mc_vocab_k: Number of vocab candidates to sample.
            closed_mc_neg_seed: Seed for deterministic negative sampling.
            closed_mc_routes: Enabled routing modes for MC loss.
            closed_use_vocab_fallback: Whether to fall back to vocab routing.
            closed_yesno_variants: Whether to use yes/no variants.
            closed_style: Prompt style for CLOSED questions.
            triples_mode: Triples mode for context injection.
            **kwargs: Keyword args forwarded to Trainer.
        """
        super().__init__(*args, **kwargs)
        self.last_batch: Dict = {}
        self.closed_vocab = list(closed_vocab or [])
        self.closed_train_objective = closed_train_objective
        self.closed_mc_weight = float(closed_mc_weight)
        self.closed_mc_max_options = int(closed_mc_max_options)
        self.closed_mc_vocab_k = int(closed_mc_vocab_k)
        self.closed_mc_neg_seed = int(closed_mc_neg_seed)
        self.closed_mc_routes = {route.strip() for route in closed_mc_routes if route.strip()}
        self.closed_use_vocab_fallback = bool(closed_use_vocab_fallback)
        self.closed_yesno_variants = bool(closed_yesno_variants)
        self.closed_style = closed_style
        self.triples_mode = triples_mode

    def _compute_closed_mc_loss(
        self,
        model: LlavaMedTrainable,
        inputs: Dict,
    ) -> tuple[torch.Tensor, Dict[str, float | int | Dict[str, int]]]:
        """Compute the multiple-choice loss for CLOSED samples.

        Args:
            model: LLaVA-Med model wrapper.
            inputs: Batch dict containing tokenized inputs and metadata.

        Returns:
            Tuple of (loss tensor, metrics dict).
        """
        answer_types = inputs.get("answer_type", [])
        questions = inputs.get("question", [])
        answers = inputs.get("answer", [])
        qids = inputs.get("qid", [])
        triples = inputs.get("triple", [])
        base_types = inputs.get("base_type", [])

        image_tensor = inputs.get("pixel_values")
        if image_tensor is None:
            return torch.zeros((), device=model.model.device), {}

        tokenizer = model.processor.tokenizer
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id or 0

        expanded_inputs: List[torch.Tensor] = []
        expanded_masks: List[torch.Tensor] = []
        expanded_prompt_lens: List[int] = []
        expanded_cand_lens: List[int] = []
        expanded_cand_ids: List[torch.Tensor] = []
        owner_indices: List[int] = []
        sample_entries: List[Dict] = []

        route_counts = {"yesno": 0, "options": 0, "vocab": 0}
        closed_count = 0

        cfg = {
            "closed_yesno_variants": self.closed_yesno_variants,
            "closed_use_vocab_fallback": self.closed_use_vocab_fallback,
        }

        for idx, answer_type in enumerate(answer_types):
            if str(answer_type).upper() != "CLOSED":
                continue
            closed_count += 1
            question = str(questions[idx])
            answer = str(answers[idx])
            qid = qids[idx] if idx < len(qids) else idx
            triple = triples[idx] if idx < len(triples) else []
            base_type = base_types[idx] if idx < len(base_types) else ""

            sample = {
                "question": question,
                "answer": answer,
                "triple": triple,
                "base_type": base_type,
            }
            candidates_info = build_closed_candidates(sample, self.closed_vocab, cfg)
            route = str(candidates_info.get("route", "vocab"))
            if route not in self.closed_mc_routes:
                continue
            labels = list(candidates_info.get("labels", []))

            if route == "yesno":
                gt_label = canonicalize_yesno(answer)
                if gt_label not in {"Yes", "No"}:
                    continue
                labels = ["Yes", "No"]
            elif route == "options":
                labels = labels[: self.closed_mc_max_options]
                target_index = find_option_target(labels, answer)
                if target_index < 0:
                    continue
                gt_label = labels[target_index]
            else:
                gt_label = answer
                if gt_label not in self.closed_vocab:
                    continue
                negatives = sample_negatives(
                    self.closed_vocab,
                    self.closed_mc_vocab_k - 1,
                    self.closed_mc_neg_seed,
                    qid,
                    gt_label,
                )
                labels = [gt_label] + negatives

            seen = set()
            deduped_labels: List[str] = []
            for label in labels:
                if label in seen:
                    continue
                seen.add(label)
                deduped_labels.append(label)
            labels = deduped_labels
            if gt_label not in labels:
                continue
            target_index = labels.index(gt_label)
            if route == "options" and len(labels) < 2:
                continue

            triples_str = build_triples_context(sample, self.triples_mode)
            user_text = build_user_text_closed(
                sample,
                options=labels if route == "options" else None,
                triples_str=triples_str,
                closed_style=self.closed_style,
            )
            prompt = build_prompt(
                conv_mode=model.conv_mode,
                user_text=user_text,
                with_image=True,
                answer_text=None,
                mm_use_im_start_end=model.mm_use_im_start_end,
            )
            prompt_ids = model.encode_prompt_with_image_token(prompt).squeeze(0)
            prompt_len = int(prompt_ids.shape[0])

            entry = {
                "indices": [],
                "target_index": target_index,
                "route": route,
                "labels": labels,
            }
            for label in labels:
                cand_ids = tokenizer(
                    label,
                    add_special_tokens=False,
                    return_tensors="pt",
                ).input_ids.squeeze(0).to(prompt_ids.device)
                cand_len = int(cand_ids.shape[0])
                if cand_len == 0:
                    continue
                seq = torch.cat([prompt_ids, cand_ids], dim=0)
                expanded_inputs.append(seq)
                expanded_masks.append(torch.ones_like(seq))
                expanded_prompt_lens.append(prompt_len)
                expanded_cand_lens.append(cand_len)
                expanded_cand_ids.append(cand_ids)
                owner_indices.append(idx)
                entry["indices"].append(len(expanded_inputs) - 1)
            if entry["indices"]:
                sample_entries.append(entry)
                route_counts[route] += 1

        if not expanded_inputs:
            empty = torch.zeros((), device=model.model.device)
            metrics = {
                "mc_loss": 0.0,
                "mc_valid_count": 0,
                "mc_closed_count": closed_count,
                "mc_route_counts": route_counts,
            }
            return empty, metrics

        input_ids = torch.nn.utils.rnn.pad_sequence(
            expanded_inputs,
            batch_first=True,
            padding_value=pad_token_id,
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            expanded_masks,
            batch_first=True,
            padding_value=0,
        )
        owner_tensor = torch.tensor(owner_indices, device=image_tensor.device)
        expanded_images = image_tensor[owner_tensor]
        target_device = input_ids.device
        target_dtype = getattr(model.model, "dtype", None)
        if target_dtype is None:
            try:
                target_dtype = next(model.model.parameters()).dtype
            except StopIteration:
                target_dtype = expanded_images.dtype
        expanded_images = expanded_images.to(device=target_device, dtype=target_dtype)
        attention_mask = attention_mask.to(device=target_device)

        if getattr(model.processor, "is_llava_official", False):
            image_kwargs = {"images": expanded_images}
        else:
            image_kwargs = {"pixel_values": expanded_images}

        outputs = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **image_kwargs,
        )
        logits = outputs.logits
        log_probs = torch.log_softmax(logits, dim=-1)

        row_scores: List[torch.Tensor] = []
        for row_idx, cand_ids in enumerate(expanded_cand_ids):
            prompt_len = expanded_prompt_lens[row_idx]
            cand_len = expanded_cand_lens[row_idx]
            total = torch.zeros((), device=log_probs.device)
            for token_offset in range(cand_len):
                token_id = cand_ids[token_offset].item()
                logit_index = prompt_len - 1 + token_offset
                total = total + log_probs[row_idx, logit_index, token_id]
            # Mean log-probability length normalization.
            score = total / max(cand_len, 1)
            row_scores.append(score)

        mc_losses: List[torch.Tensor] = []
        mc_valid_count = 0
        acc_yesno = 0
        acc_options = 0
        acc_vocab = 0
        count_yesno = 0
        count_options = 0
        count_vocab = 0
        pred_yes = 0
        pred_no = 0
        for entry in sample_entries:
            scores = torch.stack([row_scores[idx] for idx in entry["indices"]], dim=0)
            target_idx = int(entry["target_index"])
            loss = F.cross_entropy(scores.unsqueeze(0), torch.tensor([target_idx], device=scores.device))
            mc_losses.append(loss)
            mc_valid_count += 1
            pred_idx = int(torch.argmax(scores).item())
            route = entry["route"]
            if route == "yesno":
                count_yesno += 1
                if pred_idx == target_idx:
                    acc_yesno += 1
                pred_label = entry["labels"][pred_idx]
                if pred_label == "Yes":
                    pred_yes += 1
                elif pred_label == "No":
                    pred_no += 1
            elif route == "options":
                count_options += 1
                if pred_idx == target_idx:
                    acc_options += 1
            else:
                count_vocab += 1
                if pred_idx == target_idx:
                    acc_vocab += 1

        mc_loss = torch.stack(mc_losses).mean() if mc_losses else torch.zeros((), device=model.model.device)
        metrics = {
            "mc_loss": float(mc_loss.detach().item()),
            "mc_valid_count": mc_valid_count,
            "mc_closed_count": closed_count,
            "mc_route_counts": route_counts,
            "mc_acc_yesno": acc_yesno / max(count_yesno, 1),
            "mc_acc_options": acc_options / max(count_options, 1),
            "mc_acc_vocab": acc_vocab / max(count_vocab, 1),
            "mc_yesno_pred_yes": pred_yes,
            "mc_yesno_pred_no": pred_no,
        }
        return mc_loss, metrics

    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute SFT loss and optional MC loss for the batch.

        Args:
            model: Model or wrapper to invoke.
            inputs: Batch inputs from the collator.
            return_outputs: Whether to return outputs alongside loss.

        Returns:
            Loss tensor, and optionally model outputs.
        """
        extra_keys = {
            "qid",
            "answer_type",
            "question",
            "answer",
            "content_type",
            "base_type",
            "modality",
            "triple",
            "prompt_len",
            "answer_len",
            "prompt_truncated",
            "answer_truncated",
            "used_placeholder_image",
        }
        self.last_batch = {k: inputs.get(k) for k in extra_keys if k in inputs}
        model_inputs = {
            k: v
            for k, v in inputs.items()
            if k in {"input_ids", "attention_mask", "pixel_values", "labels"}
        }
        answer_types = inputs.get("answer_type", [])
        if self.closed_train_objective == "mc" and "labels" in model_inputs:
            labels = model_inputs["labels"].clone()
            for idx, answer_type in enumerate(answer_types):
                if str(answer_type).upper() == "CLOSED":
                    labels[idx] = -100
            model_inputs["labels"] = labels

        outputs = model(**model_inputs)
        sft_loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]

        mc_loss = torch.zeros((), device=sft_loss.device)
        mc_metrics = {}
        if self.closed_train_objective in {"mc", "hybrid"}:
            mc_loss, mc_metrics = self._compute_closed_mc_loss(model, inputs)

        if self.closed_train_objective == "sft":
            total_loss = sft_loss
        elif self.closed_train_objective in {"mc", "hybrid"}:
            mc_weight = torch.tensor(self.closed_mc_weight, device=sft_loss.device)
            total_loss = sft_loss + mc_weight * mc_loss
        else:
            total_loss = sft_loss

        self.last_batch.update(
            {
                "sft_loss": float(sft_loss.detach().item()),
                **mc_metrics,
            }
        )
        if return_outputs:
            return total_loss, outputs
        return total_loss


class EpochProgressCallback(TrainerCallback):
    """Callback that shows per-epoch tqdm progress bars."""
    def __init__(
        self,
        trainer: Trainer,
        enabled: bool,
        mininterval: float,
        leave: bool,
    ) -> None:
        """Initialize epoch progress reporting via tqdm."""
        self.trainer = trainer
        self.enabled = enabled
        self.mininterval = mininterval
        self.leave = leave
        self.progress_bar: Optional[tqdm] = None

    def on_epoch_begin(self, args, state, control, **kwargs):
        """Create a new progress bar at the start of each epoch.

        Args:
            args: Training arguments.
            state: Trainer state.
            control: Trainer control flags.
            **kwargs: Additional callback arguments.
        """
        if not self.enabled:
            return
        if self.progress_bar is not None:
            self.progress_bar.close()
            self.progress_bar = None
        total = getattr(self.trainer, "num_update_steps_per_epoch", None)
        if total is None:
            try:
                total = len(self.trainer.get_train_dataloader())
            except TypeError:
                total = None
        epoch_index = int(state.epoch) + 1 if state.epoch is not None else 1
        total_epochs = int(state.num_train_epochs) if state.num_train_epochs is not None else "?"
        self.progress_bar = tqdm(
            total=total,
            desc=f"Epoch {epoch_index}/{total_epochs}",
            unit="step",
            dynamic_ncols=True,
            mininterval=self.mininterval,
            leave=self.leave,
        )

    def on_step_end(self, args, state, control, **kwargs):
        """Advance the progress bar for each training step.

        Args:
            args: Training arguments.
            state: Trainer state.
            control: Trainer control flags.
            **kwargs: Additional callback arguments.
        """
        if self.progress_bar is not None:
            self.progress_bar.update(1)

    def on_epoch_end(self, args, state, control, **kwargs):
        """Close the progress bar at the end of each epoch.

        Args:
            args: Training arguments.
            state: Trainer state.
            control: Trainer control flags.
            **kwargs: Additional callback arguments.
        """
        if self.progress_bar is not None:
            self.progress_bar.close()
            self.progress_bar = None

    def on_train_end(self, args, state, control, **kwargs):
        """Close the progress bar at the end of training.

        Args:
            args: Training arguments.
            state: Trainer state.
            control: Trainer control flags.
            **kwargs: Additional callback arguments.
        """
        if self.progress_bar is not None:
            self.progress_bar.close()
            self.progress_bar = None


def set_reproducibility(seed: int) -> None:
    """Set random seeds and deterministic flags for reproducibility.

    Args:
        seed: RNG seed to apply across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic CuDNN behavior for reproducible training runs.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_git_commit() -> Optional[str]:
    """Return the current git commit hash, if available.

    Returns:
        Commit hash string, or None if unavailable.
    """
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for training.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(description="Train LLaVA-Med on SLAKE")
    parser.add_argument("--model_id", type=str, default="microsoft/llava-med-v1.5-mistral-7b")
    parser.add_argument("--fallback_model_id", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--data_source", choices=["hf", "local"], default="local")
    parser.add_argument("--slake_root", type=Path, default=Path("SLAKE"))
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--eval_split", type=str, default="validation")
    parser.add_argument("--output_dir", type=Path, default=Path("work_slake_llava_med_train"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_epochs", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_prompt_len", type=int, default=512)
    parser.add_argument("--max_answer_len", type=int, default=64)
    parser.add_argument("--use_triple_context", action="store_true")
    parser.add_argument("--triple_k", type=int, default=3)
    parser.add_argument("--max_triples", type=int, default=None)
    parser.add_argument("--conv_mode", type=str, default="mistral_instruct")
    parser.add_argument("--open_prompt_style", dest="open_style", type=str, default="short")
    parser.add_argument("--closed_prompt_style", dest="closed_style", type=str, default="minimal")
    parser.add_argument("--mask_mode", choices=["none", "crop", "masked"], default="none")
    parser.add_argument("--mask_pad_ratio", type=float, default=0.10)
    parser.add_argument("--mask_threshold", type=int, default=1)
    parser.add_argument(
        "--mask_union_mode",
        choices=["union", "prefer_disease_for_abnormality"],
        default="prefer_disease_for_abnormality",
    )
    parser.add_argument("--mask_debug_dir", type=Path, default=None)
    parser.add_argument("--mask_debug_n", type=int, default=8)
    parser.add_argument("--mask_require_hit_rate", type=float, default=0.20)
    parser.add_argument("--mask_pixel_diff_check", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--triples_mode", choices=["off", "real_only", "kvqa_real_only"], default="real_only")
    parser.add_argument("--freeze_vision_tower", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tune_mm_projector", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj",
    )
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--unfreeze_last_llm_layers", type=int, default=0)
    parser.add_argument("--logging_steps", type=int, default=25)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--resume_from_checkpoint", type=str, default="auto")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--tb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tb_logdir", type=Path, default=None)
    parser.add_argument("--tb_log_images_every", type=int, default=500)
    parser.add_argument("--tb_log_text_every", type=int, default=200)
    parser.add_argument("--tb_probe_size", type=int, default=32)
    parser.add_argument("--tb_eval_generate_size", type=int, default=128)
    parser.add_argument("--tb_hist_every", type=int, default=200)
    parser.add_argument("--max_new_tokens_open_default", type=int, default=16)
    parser.add_argument("--max_new_tokens_open_abnormality", type=int, default=48)
    parser.add_argument("--min_new_tokens_open", type=int, default=1)
    parser.add_argument("--max_new_tokens_open", type=int, default=16)
    parser.add_argument("--max_new_tokens_closed", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--do_sample", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--closed_vocab_topk", type=int, default=200)
    parser.add_argument("--closed_use_vocab_fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--closed_yesno_variants", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--closed_length_norm", choices=["mean"], default="mean")
    parser.add_argument("--closed_train_objective", choices=["sft", "mc", "hybrid"], default="hybrid")
    parser.add_argument("--closed_mc_weight", type=float, default=0.1)
    parser.add_argument("--closed_mc_max_options", type=int, default=6)
    parser.add_argument("--closed_mc_vocab_k", type=int, default=20)
    parser.add_argument("--closed_mc_neg_seed", type=int, default=0)
    parser.add_argument(
        "--closed_mc_routes",
        type=str,
        default="yesno,options",
        help="Comma-separated routes to enable for MC loss.",
    )
    tqdm_group = parser.add_mutually_exclusive_group()
    tqdm_group.add_argument("--tqdm", dest="tqdm", action="store_true")
    tqdm_group.add_argument("--no_tqdm", dest="tqdm", action="store_false")
    parser.set_defaults(tqdm=True)
    parser.add_argument("--tqdm_mininterval", type=float, default=0.1)
    parser.add_argument("--tqdm_leave", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run_final_eval", action="store_true", help="Run eval after training.")
    parser.add_argument("--final_eval_split", type=str, default="test")
    parser.add_argument("--final_eval_output_dir", type=Path, default=None)
    parser.add_argument("--final_eval_batch_size", type=int, default=None)
    return parser


def main() -> None:
    """Entry point for training LLaVA-Med on SLAKE."""
    logging.basicConfig(level=logging.INFO)
    args = build_parser().parse_args()
    args.run_final_eval = True
    logger.info(
        "Closed training objective=%s closed_mc_weight=%.3f",
        args.closed_train_objective,
        args.closed_mc_weight,
    )

    set_reproducibility(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.tb_logdir is None:
        args.tb_logdir = args.output_dir / "tb"
    if args.mask_debug_dir is None:
        args.mask_debug_dir = args.output_dir / "mask_debug"
    if args.tb:
        args.tb_logdir.mkdir(parents=True, exist_ok=True)

    records = load_slake_records(
        source=args.data_source,
        split=args.split,
        slake_root=args.slake_root,
        use_triple_context=False,
        triple_k=args.triple_k,
        max_triples=args.max_triples,
    )
    eval_records: List[Dict] = []
    if args.eval_steps > 0:
        eval_records = load_slake_records(
            source=args.data_source,
            split=args.eval_split,
            slake_root=args.slake_root,
            use_triple_context=False,
            triple_k=args.triple_k,
            max_triples=args.max_triples,
        )

    if args.dry_run:
        records = records[:5]
        eval_records = eval_records[:5]

    closed_mc_routes = [route.strip() for route in args.closed_mc_routes.split(",") if route.strip()]

    closed_vocab = build_closed_vocab_from_train(records, args.closed_vocab_topk)
    save_closed_vocab(closed_vocab, args.output_dir / "closed_vocab.json")

    train_dataset = SlakeDataset(
        records,
        args.slake_root,
        mask_mode=args.mask_mode,
        mask_pad_ratio=args.mask_pad_ratio,
        mask_threshold=args.mask_threshold,
        mask_union_mode=args.mask_union_mode,
    )
    if args.mask_mode != "none":
        hit_rate = _compute_mask_hit_rate(
            records,
            args.slake_root,
            args.mask_threshold,
            args.mask_union_mode,
            sample_limit=100,
        )
        logger.info("Mask hit-rate (train sample) = %.3f", hit_rate)
        if hit_rate < args.mask_require_hit_rate:
            raise ValueError(
                f"Mask hit-rate {hit_rate:.3f} below required threshold {args.mask_require_hit_rate:.2f}. "
                "Check mask paths/layout or set --mask_require_hit_rate."
            )
        if args.mask_debug_n > 0:
            saved = _save_mask_debug_images(
                records,
                args.slake_root,
                args.mask_mode,
                args.mask_threshold,
                args.mask_union_mode,
                args.mask_pad_ratio,
                args.mask_debug_dir,
                max_items=args.mask_debug_n,
            )
            logger.info("Saved %d mask debug samples to %s", saved, args.mask_debug_dir)
    eval_dataset = (
        SlakeDataset(
            eval_records,
            args.slake_root,
            mask_mode=args.mask_mode,
            mask_pad_ratio=args.mask_pad_ratio,
            mask_threshold=args.mask_threshold,
            mask_union_mode=args.mask_union_mode,
        )
        if eval_records
        else None
    )

    model = LlavaMedTrainable(
        model_id=args.model_id,
        fallback_model_id=args.fallback_model_id,
        conv_mode=args.conv_mode,
    )

    if args.freeze_vision_tower:
        model.freeze_vision_tower()
    if args.tune_mm_projector:
        model.set_mm_projector_trainable()
    lora_target_modules = parse_lora_target_modules(args.lora_target_modules)
    lora_summary = model.apply_lora_if_enabled(
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules_raw=args.lora_target_modules,
        lora_target_modules=lora_target_modules,
    )
    if args.unfreeze_last_llm_layers > 0:
        model.unfreeze_last_llm_layers(args.unfreeze_last_llm_layers)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    collator = LlavaSFTCollator(
        processor=model.processor,
        max_prompt_len=args.max_prompt_len,
        max_answer_len=args.max_answer_len,
        conv_mode=args.conv_mode,
        mm_use_im_start_end=model.mm_use_im_start_end,
        open_style=args.open_style,
        closed_style=args.closed_style,
        triples_mode=args.triples_mode,
    )
    if args.mask_mode != "none" and args.mask_pixel_diff_check:
        no_mask_dataset = SlakeDataset(
            records,
            args.slake_root,
            mask_mode="none",
            mask_pad_ratio=args.mask_pad_ratio,
            mask_threshold=args.mask_threshold,
            mask_union_mode=args.mask_union_mode,
        )
        indices = _select_diff_indices(len(train_dataset))
        diff = _run_mask_pixel_diff_check(model, train_dataset, no_mask_dataset, indices)
        logger.info("Mask pixel diff check mean_abs_diff=%.6f", diff)
        if diff <= 1e-4:
            raise ValueError(
                "Mask pixel diff check failed: masked and unmasked pixel_values are too similar. "
                "Verify mask resolution and --mask_mode."
            )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_epochs,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        eval_steps=args.eval_steps,
        evaluation_strategy="steps" if eval_dataset is not None else "no",
        save_total_limit=args.save_total_limit,
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        remove_unused_columns=False,
        report_to=["tensorboard"] if args.tb else [],
        logging_dir=str(args.tb_logdir),
        dataloader_drop_last=False,
        max_steps=2 if args.dry_run else -1,
        disable_tqdm=True,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    resume_checkpoint = None
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint == "auto":
            resume_checkpoint = get_last_checkpoint(str(args.output_dir))
        else:
            resume_checkpoint = args.resume_from_checkpoint

    trainer = VQATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        closed_vocab=closed_vocab,
        closed_train_objective=args.closed_train_objective,
        closed_mc_weight=args.closed_mc_weight,
        closed_mc_max_options=args.closed_mc_max_options,
        closed_mc_vocab_k=args.closed_mc_vocab_k,
        closed_mc_neg_seed=args.closed_mc_neg_seed,
        closed_mc_routes=closed_mc_routes,
        closed_use_vocab_fallback=args.closed_use_vocab_fallback,
        closed_yesno_variants=args.closed_yesno_variants,
        closed_style=args.closed_style,
        triples_mode=args.triples_mode,
    )
    assert not hasattr(model, "mc_weight")
    trainer.add_callback(
        EpochProgressCallback(
            trainer=trainer,
            enabled=args.tqdm,
            mininterval=args.tqdm_mininterval,
            leave=args.tqdm_leave,
        )
    )
    if args.tb:
        probe_records = select_probe(records, size=args.tb_probe_size, seed=args.seed)
        for record in tqdm(
            probe_records,
            desc="Preparing probe samples",
            unit="sample",
            dynamic_ncols=True,
            mininterval=args.tqdm_mininterval,
            leave=args.tqdm_leave,
            disable=False,
        ):
            record["image_path"] = str(resolve_image_path(args.slake_root, record["img_name"]))
        tb_writer = SafeSummaryWriter(log_dir=str(args.tb_logdir))
        trainer.add_callback(
            VQATensorBoardCallback(
                writer=tb_writer,
                trainer=trainer,
                probe_records=probe_records,
                closed_vocab=closed_vocab,
                slake_root=args.slake_root,
                tb_log_images_every=args.tb_log_images_every,
                tb_log_text_every=args.tb_log_text_every,
                tb_eval_generate_size=args.tb_eval_generate_size,
                tb_hist_every=args.tb_hist_every,
                open_min_new_tokens=args.min_new_tokens_open,
                mask_mode=args.mask_mode,
                mask_pad_ratio=args.mask_pad_ratio,
                mask_threshold=args.mask_threshold,
                mask_union_mode=args.mask_union_mode,
                triples_mode=args.triples_mode,
                open_style=args.open_style,
                closed_style=args.closed_style,
                closed_yesno_variants=args.closed_yesno_variants,
            )
        )

    trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_model()

    run_config = vars(args).copy()
    run_config["git_commit"] = get_git_commit()
    run_config["lora_target_modules_raw"] = args.lora_target_modules
    run_config["lora_target_modules_parsed"] = lora_target_modules
    if lora_summary is not None:
        run_config["lora_summary"] = lora_summary
    run_config_path = args.output_dir / "run_config.json"
    with run_config_path.open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2, default=str)

    if args.run_final_eval:
        from eval_llava_med import build_parser as build_eval_parser
        from eval_llava_med import run_evaluation

        eval_output_dir = args.final_eval_output_dir or args.output_dir / "eval"
        eval_parser = build_eval_parser()
        eval_args = eval_parser.parse_args(
            [
                "--model_id_or_ckpt",
                str(args.output_dir),
                "--data_source",
                args.data_source,
                "--slake_root",
                str(args.slake_root),
                "--split",
                args.final_eval_split,
                "--output_dir",
                str(eval_output_dir),
                "--conv_mode",
                args.conv_mode,
                "--triple_k",
                str(args.triple_k),
                "--min_new_tokens_open",
                str(args.min_new_tokens_open),
                "--max_new_tokens_open_default",
                str(args.max_new_tokens_open_default),
                "--max_new_tokens_open_abnormality",
                str(args.max_new_tokens_open_abnormality),
                "--max_new_tokens_closed",
                str(args.max_new_tokens_closed),
                "--temperature",
                str(args.temperature),
                "--do_sample" if args.do_sample else "--no-do_sample",
            ]
            + [
                "--open_style",
                args.open_style,
                "--closed_style",
                args.closed_style,
                "--mask_mode",
                args.mask_mode,
                "--mask_pad_ratio",
                str(args.mask_pad_ratio),
                "--mask_threshold",
                str(args.mask_threshold),
                "--mask_union_mode",
                args.mask_union_mode,
                "--triples_mode",
                args.triples_mode,
                "--closed_vocab_topk",
                str(args.closed_vocab_topk),
                "--closed_use_vocab_fallback"
                if args.closed_use_vocab_fallback
                else "--no-closed_use_vocab_fallback",
                "--closed_yesno_variants"
                if args.closed_yesno_variants
                else "--no-closed_yesno_variants",
                "--closed_length_norm",
                args.closed_length_norm,
            ]
            + (["--max_triples", str(args.max_triples)] if args.max_triples is not None else [])
        )
        eval_args.batch_size = args.final_eval_batch_size or args.batch_size
        eval_args.tqdm = args.tqdm
        eval_args.tqdm_mininterval = args.tqdm_mininterval
        eval_args.tqdm_leave = args.tqdm_leave
        eval_args.seed = args.seed
        eval_args.dry_run = args.dry_run
        eval_args.allow_build_vocab = True
        run_evaluation(eval_args, model=model)


if __name__ == "__main__":
    main()
