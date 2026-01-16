"""Evaluate LLaVA-Med on SLAKE with separated OPEN/CLOSED handling."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from tqdm.auto import tqdm

from data.slake_dataset import load_slake_records, resolve_image_path
from models.llava_med_trainable import LlavaMedTrainable
from utils.closed_candidates import (
    is_yesno_question,
    make_candidate_variants,
    parse_options_from_question,
    select_topk_vocab_candidates,
)
from utils.closed_router import (
    build_closed_candidates,
    build_closed_vocab_from_train,
    has_option_markers,
    load_closed_vocab,
    save_closed_vocab,
)
from utils.generation_constraints import get_open_bad_words_ids
from utils.mask_preprocess import (
    apply_mask,
    load_mask,
    load_segmentation_map,
    resolve_segmentation_mask_path,
)
from utils.prompting import build_triples_context, build_user_text_open, resolve_stop_strings
from utils.tensorboard_utils import SafeSummaryWriter, format_kv_table
from utils.text_norm import extract_first_non_empty_line, normalize_answer, open_token_f1

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
    """Create a visualization image for segmentation or binary masks."""
    if seg_map is None:
        resized_mask = _resize_mask_to_image(mask, image)
        vis = resized_mask.astype("uint8") * 255
    else:
        seg_map = np.array(Image.fromarray(seg_map.astype("uint8")).resize(image.size, resample=Image.NEAREST))
        max_val = float(seg_map.max()) if seg_map.size else 0.0
        if max_val > 0:
            vis = (seg_map.astype("float32") / max_val * 255).astype("uint8")
        else:
            vis = seg_map.astype("uint8")
    return Image.fromarray(vis)


def _build_overlay(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """Overlay a red mask on top of an image for visualization."""
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
    """Compute the fraction of samples with available masks."""
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
    """Save mask debug visualizations to disk."""
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


def set_reproducibility(seed: int) -> None:
    """Set random seeds and deterministic flags for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic CuDNN behavior for reproducible evaluation.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_gen_settings_hash(settings: Dict) -> str:
    """Compute a stable hash for generation settings.

    Args:
        settings: Generation settings mapping.

    Returns:
        MD5 hash string for the settings payload.
    """
    payload = json.dumps(settings, sort_keys=True)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def to_json_compatible(value):
    """Convert objects into JSON-serializable Python types.

    Args:
        value: Input value to convert.

    Returns:
        JSON-serializable representation.
    """
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: to_json_compatible(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_compatible(item) for item in value]
    return value


def load_image(
    sample: Dict,
    slake_root: Path,
    mask_mode: str,
    mask_threshold: int,
    mask_union_mode: str,
    mask_pad_ratio: float,
    use_tqdm: bool = False,
) -> Optional[Image.Image]:
    """Load an image and apply masks if configured.

    Args:
        sample: Dataset record containing ``img_name``.
        slake_root: SLAKE dataset root directory.
        mask_mode: Masking mode (``none``, ``masked``, ``crop``).
        mask_threshold: Threshold for binary masks.
        mask_union_mode: Mask union strategy.
        mask_pad_ratio: Padding ratio for crop masks.
        use_tqdm: Whether to log warnings via tqdm.

    Returns:
        Loaded PIL image, or ``None`` if loading fails.
    """
    image_path = resolve_image_path(slake_root, sample["img_name"])
    try:
        with Image.open(image_path) as img:
            image = img.convert("RGB")
    except (FileNotFoundError, UnidentifiedImageError) as exc:
        if use_tqdm:
            tqdm.write(f"Invalid image {image_path}: {exc}")
        else:
            logger.warning("Invalid image %s: %s", image_path, exc)
        return None
    if mask_mode != "none":
        mask = load_mask(sample, slake_root, mask_threshold, mask_union_mode)
        image = apply_mask(image, mask, mask_mode, mask_pad_ratio)
    return image


def build_gt_record(sample: Dict, normalized_gt: str) -> Dict:
    """Build a JSONL ground-truth record for artifact writing.

    Args:
        sample: Dataset record.
        normalized_gt: Normalized ground-truth answer.

    Returns:
        Record dict including normalized and raw ground truth.
    """
    record = dict(sample)
    record["normalized_gt"] = normalized_gt
    record["raw_gt"] = sample.get("answer", "")
    return record


def update_metrics(
    metrics: Dict,
    sample: Dict,
    normalized_pred: str,
    normalized_gt: str,
    open_f1: Optional[float] = None,
) -> None:
    """Update running metric totals for a single sample.

    Args:
        metrics: Metrics accumulator dict.
        sample: Dataset record.
        normalized_pred: Normalized prediction string.
        normalized_gt: Normalized ground-truth string.
        open_f1: Optional token-F1 for OPEN answers.
    """
    is_correct = normalized_pred == normalized_gt
    key = sample["answer_type"].upper()
    metrics["correct_counts"][key] += int(is_correct)
    metrics["total_counts"][key] += 1
    metrics["correct_counts"]["overall"] += int(is_correct)
    metrics["total_counts"]["overall"] += 1

    for subgroup in ["answer_type", "modality", "location", "q_lang", "content_type", "base_type"]:
        metrics["subgroup_correct"][subgroup][sample.get(subgroup, "")] += int(is_correct)
        metrics["subgroup_total"][subgroup][sample.get(subgroup, "")] += 1
        if key == "OPEN" and open_f1 is not None:
            metrics["subgroup_open_f1_sum"][subgroup][sample.get(subgroup, "")] += open_f1
            metrics["subgroup_open_count"][subgroup][sample.get(subgroup, "")] += 1

    if key == "OPEN" and open_f1 is not None:
        metrics["open_f1_sum"] += open_f1
        metrics["open_count"] += 1


def finalize_metrics(metrics: Dict) -> Dict:
    """Finalize aggregate metrics into normalized rates.

    Args:
        metrics: Metrics accumulator dict.

    Returns:
        Metrics dict with aggregate scores and subgroup breakdowns.
    """
    open_token_f1_mean = metrics["open_f1_sum"] / metrics["open_count"] if metrics["open_count"] else 0.0
    closed_acc = (
        metrics["correct_counts"]["CLOSED"] / metrics["total_counts"]["CLOSED"]
        if metrics["total_counts"]["CLOSED"]
        else 0.0
    )
    output = {
        "open/token_f1": open_token_f1_mean,
        "open/exact_match": (
            metrics["correct_counts"]["OPEN"] / metrics["total_counts"]["OPEN"]
            if metrics["total_counts"]["OPEN"]
            else 0.0
        ),
        "closed/accuracy": closed_acc,
        "overall/score": 0.5 * closed_acc + 0.5 * open_token_f1_mean,
        "overall/accuracy": (
            metrics["correct_counts"]["overall"] / metrics["total_counts"]["overall"]
            if metrics["total_counts"]["overall"]
            else 0.0
        ),
        "counts": dict(metrics["total_counts"]),
        "invalid_image": metrics["invalid_image"],
        "subgroups": {},
    }

    for subgroup in metrics["subgroup_total"]:
        totals = metrics["subgroup_total"][subgroup]
        corrects = metrics["subgroup_correct"][subgroup]
        f1_sum = metrics["subgroup_open_f1_sum"][subgroup]
        open_counts = metrics["subgroup_open_count"][subgroup]
        output["subgroups"][subgroup] = {
            "accuracy": {key: (corrects[key] / totals[key]) if totals[key] else 0.0 for key in totals},
            "open_token_f1": {
                key: (f1_sum[key] / open_counts[key]) if open_counts[key] else 0.0 for key in totals
            },
            "counts": dict(totals),
        }

    return output


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    """Write rows to a JSONL file.

    Args:
        path: Output file path.
        rows: Iterable of row dicts.
    """
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_git_commit() -> Optional[str]:
    """Return the current git commit hash, if available.

    Returns:
        Commit hash string, or None if unavailable.
    """
    try:
        import subprocess

        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception:
        return None


def is_rank_zero() -> bool:
    """Return True if running on the main process for distributed eval.

    Returns:
        True if current process is rank zero or distributed is disabled.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for evaluation.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(description="Evaluate LLaVA-Med on SLAKE")
    parser.add_argument("--model_id_or_ckpt", type=str, required=True)
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--data_source", choices=["hf", "local"], default="local")
    parser.add_argument("--slake_root", type=Path, default=Path("SLAKE"))
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output_dir", type=Path, default=Path("work_slake_llava_med_eval"))
    parser.add_argument("--use_triple_context", action="store_true")
    parser.add_argument("--triple_k", type=int, default=3)
    parser.add_argument("--max_triples", type=int, default=None)
    parser.add_argument("--conv_mode", type=str, default="mistral_instruct")
    parser.add_argument("--open_style", type=str, default="short")
    parser.add_argument("--closed_style", type=str, default="minimal")
    parser.add_argument("--mask_mode", choices=["none", "crop", "masked"], default="none")
    parser.add_argument("--mask_pad_ratio", type=float, default=0.10)
    parser.add_argument("--mask_threshold", type=int, default=1)
    parser.add_argument(
        "--mask_union_mode",
        choices=["union", "prefer_disease_for_abnormality"],
        default="prefer_disease_for_abnormality",
    )
    parser.add_argument("--mask_debug_dir", type=Path, default=None)
    parser.add_argument("--mask_debug_n", type=int, default=4)
    parser.add_argument("--mask_warn_hit_rate", type=float, default=0.05)
    parser.add_argument("--triples_mode", choices=["off", "real_only", "kvqa_real_only"], default="real_only")
    parser.add_argument("--mode", choices=["eval", "diagnose"], default="eval")
    parser.add_argument("--probe_size", type=int, default=200)
    parser.add_argument("--probe_seed", type=int, default=0)
    parser.add_argument("--probe_split", choices=["val", "test"], default="test")
    parser.add_argument("--diagnose_out_dir", type=Path, default=None)
    parser.add_argument("--diagnose_examples", type=int, default=25)
    parser.add_argument(
        "--diagnose_generic_set",
        type=str,
        default="image,picture,photo,scan,xray,x-ray",
    )
    parser.add_argument(
        "--diagnose_yes_variants",
        type=str,
        default="Yes,yes, yes,Yes., yes.,YES",
    )
    parser.add_argument(
        "--diagnose_no_variants",
        type=str,
        default="No,no, no,No., no.,NO",
    )
    parser.add_argument("--max_new_tokens_open_default", type=int, default=16)
    parser.add_argument("--max_new_tokens_open_abnormality", type=int, default=48)
    parser.add_argument("--min_new_tokens_open", type=int, default=1)
    parser.add_argument("--max_new_tokens_open", type=int, default=16)
    parser.add_argument("--max_new_tokens_closed", type=int, default=4)
    parser.add_argument("--min_new_tokens", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--do_sample", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--open_stop_strings", type=str, default=None)
    parser.add_argument("--smoke_test", action="store_true", default=False)
    parser.add_argument("--smoke_test_samples", type=int, default=10)
    parser.add_argument("--closed_candidate_mode", choices=["yesno", "parse_options", "vocab", "auto"], default="auto")
    parser.add_argument("--closed_vocab_path", type=Path, default=None)
    parser.add_argument("--allow_build_vocab", action="store_true")
    parser.add_argument("--topk_vocab_candidates", type=int, default=50)
    parser.add_argument("--closed_vocab_topk", type=int, default=200)
    parser.add_argument("--closed_use_vocab_fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--closed_yesno_variants", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--closed_length_norm", choices=["mean"], default="mean")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--tb", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tb_logdir", type=Path, default=None)
    parser.add_argument("--tb_run_name", type=str, default=None)
    tqdm_group = parser.add_mutually_exclusive_group()
    tqdm_group.add_argument("--tqdm", dest="tqdm", action="store_true")
    tqdm_group.add_argument("--no_tqdm", dest="tqdm", action="store_false")
    parser.set_defaults(tqdm=True)
    parser.add_argument("--tqdm_mininterval", type=float, default=0.1)
    parser.add_argument("--tqdm_leave", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tqdm_nested_candidates", action="store_true", default=False)
    return parser


def parse_csv_list(value: str) -> List[str]:
    """Parse a comma-separated list into a list of strings.

    Args:
        value: Comma-separated string.

    Returns:
        List of non-empty items.
    """
    items = [item.strip() for item in (value or "").split(",")]
    return [item for item in items if item]


def get_open_max_new_tokens(content_type: str, args: argparse.Namespace) -> int:
    """Select max_new_tokens for OPEN answers based on content type.

    Args:
        content_type: Content type string from the dataset.
        args: Parsed evaluation args containing token limits.

    Returns:
        Maximum number of new tokens for OPEN generation.
    """
    if str(content_type or "") == "Abnormality":
        return int(args.max_new_tokens_open_abnormality)
    return int(args.max_new_tokens_open_default)


def strip_answer_prefix(text: str) -> str:
    """Strip common Answer: prefixes from text.

    Args:
        text: Raw model output.

    Returns:
        Cleaned answer string with prefixes removed.
    """
    cleaned = extract_first_non_empty_line(text)
    if cleaned.lower().startswith("answer:"):
        return cleaned.split(":", 1)[1].strip()
    return cleaned


def is_image_index_answer(ans: str) -> bool:
    """Return True if answer appears to reference an image index.

    Args:
        ans: Answer string to inspect.

    Returns:
        True if the answer looks like "Image N".
    """
    cleaned = ans.strip().lower()
    return re.fullmatch(r"image\s*\d+", cleaned) is not None


def truncate_text(text: str, max_len: int = 400) -> str:
    """Truncate text to a max length with ellipsis.

    Args:
        text: Input text string.
        max_len: Maximum length before truncation.

    Returns:
        Possibly truncated text with ellipsis.
    """
    if len(text) <= max_len:
        return text
    return f"{text[: max_len - 3]}..."


def sample_probe_records(records: List[Dict], size: int, seed: int) -> List[Dict]:
    """Sample a deterministic probe subset of records.

    Args:
        records: Dataset records list.
        size: Desired sample size.
        seed: RNG seed for determinism.

    Returns:
        List of sampled records.
    """
    if size <= 0 or not records:
        return []
    rng = random.Random(seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)
    selected = indices[: min(size, len(indices))]
    return [records[idx] for idx in selected]


def select_smoke_samples(records: List[Dict], count: int, seed: int) -> List[Dict]:
    """Select a deterministic subset for smoke testing.

    Args:
        records: Dataset records list.
        count: Total number of samples to select.
        seed: RNG seed for determinism.

    Returns:
        List of selected records, balanced by OPEN/CLOSED when possible.
    """
    rng = random.Random(seed)
    open_samples = [rec for rec in records if rec.get("answer_type", "").upper() == "OPEN"]
    closed_samples = [rec for rec in records if rec.get("answer_type", "").upper() == "CLOSED"]
    rng.shuffle(open_samples)
    rng.shuffle(closed_samples)
    target_open = min(len(open_samples), count // 2)
    target_closed = min(len(closed_samples), count - target_open)
    selected = open_samples[:target_open] + closed_samples[:target_closed]
    if len(selected) < count:
        remaining = open_samples[target_open:] + closed_samples[target_closed:]
        rng.shuffle(remaining)
        selected.extend(remaining[: count - len(selected)])
    rng.shuffle(selected)
    return selected


def run_smoke_test(
    args: argparse.Namespace,
    model: LlavaMedTrainable,
    records: List[Dict],
) -> Dict:
    """Run a minimal smoke test for OPEN/CLOSED pipelines.

    Args:
        args: Parsed evaluation args.
        model: LLaVA-Med model wrapper.
        records: Dataset records list.

    Returns:
        Dict of smoke-test statistics.
    """
    smoke_samples = select_smoke_samples(records, int(args.smoke_test_samples), args.seed)
    if args.open_stop_strings:
        stop_strings = [s for s in args.open_stop_strings.split(",") if s]
    else:
        stop_strings = resolve_stop_strings(args.conv_mode)
    non_empty_count = 0
    token_lens: List[int] = []
    for sample in smoke_samples:
        image = load_image(
            sample,
            args.slake_root,
            args.mask_mode,
            args.mask_threshold,
            args.mask_union_mode,
            args.mask_pad_ratio,
            use_tqdm=False,
        )
        if image is None:
            image = Image.new("RGB", (224, 224), color=0)
        triples_str = build_triples_context(sample, args.triples_mode)
        user_text = build_user_text_open(
            sample,
            triples_str=triples_str,
            open_style=args.open_style,
        )
        prompt = model.build_prompt(
            user_text=user_text,
            answer_text=None,
            add_generation_prompt=True,
        )
        max_new_tokens = (
            args.max_new_tokens_closed
            if sample.get("answer_type", "").upper() == "CLOSED"
            else get_open_max_new_tokens(sample.get("content_type", ""), args)
        )
        outputs, metadata = model.generate_open(
            [image],
            [user_text],
            max_new_tokens=max_new_tokens,
            stop_strings=stop_strings,
            return_metadata=True,
            min_new_tokens=args.min_new_tokens_open,
            temperature=args.temperature,
            do_sample=args.do_sample,
        )
        output = outputs[0]
        meta = metadata[0] if metadata else {}
        token_lens.append(int(meta.get("generated_token_len", 0)))
        extracted = output.get("extracted_text", "")
        if extracted.strip():
            non_empty_count += 1
        logger.info(
            "SMOKE type=%s prompt=%s",
            sample.get("answer_type"),
            truncate_text(prompt, max_len=200),
        )
        logger.info(
            "SMOKE raw=%r extracted=%r",
            output.get("raw_text"),
            extracted,
        )
    avg_gen_len = sum(token_lens) / max(1, len(token_lens))
    logger.info("SMOKE avg_generated_token_len=%.2f", avg_gen_len)
    min_non_empty = max(1, len(smoke_samples) - 1)
    if non_empty_count < min_non_empty:
        raise RuntimeError(
            f"Smoke test failed: non-empty outputs {non_empty_count}/{len(smoke_samples)}"
        )
    return {
        "smoke_samples": len(smoke_samples),
        "non_empty_outputs": non_empty_count,
        "avg_generated_token_len": avg_gen_len,
    }


def run_evaluation(args: argparse.Namespace, model: Optional[LlavaMedTrainable] = None) -> Dict:
    """Run full evaluation over SLAKE records.

    Args:
        args: Parsed CLI arguments.
        model: Optional pre-loaded LLaVA-Med model wrapper.

    Returns:
        Metrics dict containing aggregate evaluation results.
    """
    set_reproducibility(args.seed)
    use_tqdm = True
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.tb_logdir is None:
        args.tb_logdir = args.output_dir / "tb"
    if args.mask_debug_dir is None:
        args.mask_debug_dir = args.output_dir / "mask_debug"

    records = load_slake_records(
        source=args.data_source,
        split=args.split,
        slake_root=args.slake_root,
        use_triple_context=False,
        triple_k=args.triple_k,
        max_triples=args.max_triples,
    )

    mask_hit_rate: Optional[float] = None
    mask_hit_rate_samples = 0
    mask_hit_rate_warning: Optional[str] = None
    if args.mask_mode != "none":
        mask_hit_rate_samples = min(len(records), 50)
        mask_hit_rate = _compute_mask_hit_rate(
            records,
            args.slake_root,
            args.mask_threshold,
            args.mask_union_mode,
            sample_limit=mask_hit_rate_samples,
        )
        logger.info("Mask hit-rate (eval sample) = %.3f", mask_hit_rate)
        if mask_hit_rate_samples > 0 and mask_hit_rate < args.mask_warn_hit_rate:
            mask_hit_rate_warning = (
                f"Mask hit-rate {mask_hit_rate:.3f} below warn threshold {args.mask_warn_hit_rate:.2f}. "
                "Check mask paths/layout."
            )
            logger.warning(mask_hit_rate_warning)
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
    if args.dry_run:
        records = records[:5]

    closed_vocab: List[str] = []
    if args.closed_vocab_path:
        closed_vocab = load_closed_vocab(args.closed_vocab_path)
    else:
        ckpt_dir = Path(args.model_id_or_ckpt)
        default_vocab_path = ckpt_dir / "closed_vocab.json"
        if default_vocab_path.exists():
            closed_vocab = load_closed_vocab(default_vocab_path)
        elif args.allow_build_vocab:
            train_records = load_slake_records(
                source=args.data_source,
                split="train",
                slake_root=args.slake_root,
                use_triple_context=False,
                triple_k=args.triple_k,
                max_triples=args.max_triples,
            )
            closed_vocab = build_closed_vocab_from_train(train_records, args.closed_vocab_topk)
            save_closed_vocab(closed_vocab, args.output_dir / "closed_vocab.json")
        elif args.closed_use_vocab_fallback:
            raise RuntimeError(
                "closed_vocab.json missing; provide --closed_vocab_path or enable --allow_build_vocab"
            )

    if model is None:
        model = LlavaMedTrainable(
            model_id=args.model_id_or_ckpt,
            model_base=args.model_base,
            conv_mode=args.conv_mode,
        )
    model.model.eval()

    if args.smoke_test:
        return run_smoke_test(args, model, records)

    metrics = {
        "correct_counts": Counter(),
        "total_counts": Counter(),
        "subgroup_correct": defaultdict(Counter),
        "subgroup_total": defaultdict(Counter),
        "subgroup_open_f1_sum": defaultdict(Counter),
        "subgroup_open_count": defaultdict(Counter),
        "open_f1_sum": 0.0,
        "open_count": 0,
        "invalid_image": 0,
    }

    gt_records: List[Dict] = []
    pred_records: List[Dict] = []
    open_output_lens: List[int] = []
    open_extracted_lens: List[int] = []
    closed_margins: List[float] = []
    closed_candidate_counts: List[int] = []
    closed_mode_counts = Counter()
    closed_option_gold_hits = 0
    closed_option_total = 0
    misrouted_option_to_yesno_count = 0
    qual_samples: List[Dict] = []
    open_empty_raw = 0
    open_empty_extracted = 0
    open_image_index_count = 0
    open_image_index_retry_count = 0
    open_prediction_counts: Counter[str] = Counter()
    last_prompt_token_counts: Counter[int] = Counter()
    first_new_token_counts: Counter[int] = Counter()

    if args.open_stop_strings:
        open_stop_strings = [s for s in args.open_stop_strings.split(",") if s]
    else:
        open_stop_strings = resolve_stop_strings(args.conv_mode)
    gen_settings = {
        "open_max_new_tokens_default": args.max_new_tokens_open_default,
        "open_max_new_tokens_abnormality": args.max_new_tokens_open_abnormality,
        "open_min_new_tokens": args.min_new_tokens_open,
        "open_stop_strings": open_stop_strings,
        "temperature": args.temperature,
        "do_sample": args.do_sample,
    }
    gen_hash = compute_gen_settings_hash(gen_settings)

    open_batch_images: List[Image.Image] = []
    open_batch_user_texts: List[str] = []
    open_batch_samples: List[Dict] = []
    open_batch_max_tokens: Optional[int] = None
    open_count = 0
    closed_count = 0
    invalid_count = 0
    def flush_open_batch() -> None:
        nonlocal open_empty_raw, open_empty_extracted
        nonlocal open_image_index_count, open_image_index_retry_count, open_prediction_counts
        nonlocal open_batch_max_tokens
        if not open_batch_images:
            return
        max_new_tokens = open_batch_max_tokens or args.max_new_tokens_open_default
        outputs, metadata = model.generate_open(
            open_batch_images,
            open_batch_user_texts,
            max_new_tokens=max_new_tokens,
            stop_strings=open_stop_strings,
            return_metadata=True,
            min_new_tokens=args.min_new_tokens_open,
            temperature=args.temperature,
            do_sample=args.do_sample,
        )
        image_index_indices = []
        for idx, output in enumerate(outputs):
            extracted_pred = strip_answer_prefix(output["extracted_text"])
            if is_image_index_answer(extracted_pred):
                image_index_indices.append(idx)
        if image_index_indices:
            retry_images = [open_batch_images[idx] for idx in image_index_indices]
            retry_texts = [open_batch_user_texts[idx] for idx in image_index_indices]
            bad_words_ids = get_open_bad_words_ids(model.processor.tokenizer)
            retry_outputs, retry_metadata = model.generate_open(
                retry_images,
                retry_texts,
                max_new_tokens=max_new_tokens,
                stop_strings=open_stop_strings,
                return_metadata=True,
                min_new_tokens=args.min_new_tokens_open,
                temperature=args.temperature,
                do_sample=args.do_sample,
                bad_words_ids=bad_words_ids,
            )
            for idx, output, meta in zip(image_index_indices, retry_outputs, retry_metadata):
                outputs[idx] = output
                metadata[idx] = meta
            open_image_index_retry_count += len(image_index_indices)
        for sample, output, meta, user_text in zip(
            open_batch_samples, outputs, metadata, open_batch_user_texts
        ):
            raw_generation = output["raw_text"]
            extracted_pred = strip_answer_prefix(output["extracted_text"])
            normalized_pred = normalize_answer(extracted_pred)
            normalized_gt = normalize_answer(sample["answer"])
            open_f1 = open_token_f1(extracted_pred, sample["answer"])
            open_output_lens.append(int(meta["generated_token_len"]))
            open_extracted_lens.append(len(extracted_pred.split()))
            if not raw_generation.strip():
                open_empty_raw += 1
            if not extracted_pred.strip():
                open_empty_extracted += 1
            if extracted_pred.strip():
                open_prediction_counts[extracted_pred.strip().lower()] += 1
            if is_image_index_answer(extracted_pred):
                open_image_index_count += 1
            first_new_token_id = meta.get("first_new_token_id")
            if isinstance(first_new_token_id, int):
                first_new_token_counts[first_new_token_id] += 1
            prompt = model.build_prompt(
                user_text=user_text,
                answer_text=None,
                add_generation_prompt=True,
            )
            prompt_ids = model.processor.tokenizer(prompt, add_special_tokens=False).input_ids
            if prompt_ids:
                last_prompt_token_counts[int(prompt_ids[-1])] += 1
            pred_records.append(
                {
                    "qid": sample["qid"],
                    "raw_generation": raw_generation,
                    "extracted_pred": extracted_pred,
                    "normalized_pred": normalized_pred,
                    "answer_type": sample["answer_type"],
                    "gen_settings_hash": gen_hash,
                    "invalid_image": False,
                }
            )
            if len(qual_samples) < 4:
                qual_samples.append(
                    {
                        "image_path": sample.get("image_path"),
                        "question": sample.get("question"),
                        "answer_type": sample.get("answer_type"),
                        "gt": sample.get("answer"),
                        "pred": raw_generation,
                    }
                )
            update_metrics(metrics, sample, normalized_pred, normalized_gt, open_f1=open_f1)
        open_batch_images.clear()
        open_batch_user_texts.clear()
        open_batch_samples.clear()
        open_batch_max_tokens = None

    pbar = tqdm(
        records,
        desc=f"Eval {args.split}",
        unit="sample",
        dynamic_ncols=True,
        mininterval=args.tqdm_mininterval,
        leave=args.tqdm_leave,
        disable=False,
    )
    for sample in pbar:
        image_path = resolve_image_path(args.slake_root, sample["img_name"])
        sample["image_path"] = str(image_path)
        image = load_image(
            sample,
            args.slake_root,
            args.mask_mode,
            args.mask_threshold,
            args.mask_union_mode,
            args.mask_pad_ratio,
            use_tqdm=use_tqdm,
        )
        normalized_gt = normalize_answer(sample["answer"])
        gt_records.append(build_gt_record(sample, normalized_gt))
        if image is None:
            metrics["invalid_image"] += 1
            invalid_count += 1
            pred_records.append(
                {
                    "qid": sample["qid"],
                    "raw_generation": "",
                    "extracted_pred": "",
                    "normalized_pred": "",
                    "answer_type": sample["answer_type"],
                    "gen_settings_hash": gen_hash,
                    "invalid_image": True,
                }
            )
            pbar.set_postfix_str(
                f"OPEN={open_count} CLOSED={closed_count} invalid={invalid_count}"
            )
            continue

        if sample["answer_type"].upper() == "OPEN":
            open_count += 1
            triples_str = build_triples_context(sample, args.triples_mode)
            user_text = build_user_text_open(
                sample,
                triples_str=triples_str,
                open_style=args.open_style,
            )
            max_tokens = get_open_max_new_tokens(sample.get("content_type", ""), args)
            if open_batch_max_tokens is None:
                open_batch_max_tokens = max_tokens
            if open_batch_max_tokens != max_tokens:
                flush_open_batch()
                open_batch_max_tokens = max_tokens
            open_batch_images.append(image)
            open_batch_user_texts.append(user_text)
            open_batch_samples.append(sample)
            if len(open_batch_images) >= args.batch_size:
                flush_open_batch()
            pbar.set_postfix_str(
                f"OPEN={open_count} CLOSED={closed_count} invalid={invalid_count}"
            )
            continue

        flush_open_batch()
        closed_count += 1
        cfg = {
            "closed_yesno_variants": args.closed_yesno_variants,
            "closed_use_vocab_fallback": args.closed_use_vocab_fallback,
            "closed_style": args.closed_style,
            "triples_mode": args.triples_mode,
        }
        candidates_info = build_closed_candidates(sample, closed_vocab, cfg)
        routed = str(candidates_info.get("route", "vocab"))
        if routed == "options":
            closed_option_total += 1
            candidates_normalized = [
                normalize_answer(label) for label in candidates_info.get("labels", [])
            ]
            if normalized_gt in candidates_normalized:
                closed_option_gold_hits += 1
        if routed == "yesno" and has_option_markers(sample.get("question", "")):
            misrouted_option_to_yesno_count += 1
        best_label, debug = model.score_closed(sample, image, closed_vocab, cfg)
        if not best_label:
            raise RuntimeError("No CLOSED candidates available after applying routing rules")
        normalized_pred = normalize_answer(best_label)
        closed_candidate_counts.append(int(debug.get("candidate_count", 0)))
        closed_mode_counts[debug.get("route", "vocab")] += 1
        closed_margins.append(float(debug.get("margin", 0.0)))
        pred_records.append(
            {
                "qid": sample["qid"],
                "route": debug.get("route"),
                "chosen_variant": debug.get("chosen_variant"),
                "best_score": debug.get("best_score"),
                "second_score": debug.get("second_score"),
                "margin": debug.get("margin"),
                "normalized_pred": normalized_pred,
                "answer_type": sample["answer_type"],
                "gen_settings_hash": gen_hash,
                "invalid_image": False,
            }
        )
        if len(qual_samples) < 4:
            qual_samples.append(
                {
                    "image_path": str(image_path),
                    "question": sample.get("question"),
                    "answer_type": sample.get("answer_type"),
                    "gt": sample.get("answer"),
                    "pred": best_label,
                }
            )
        update_metrics(metrics, sample, normalized_pred, normalized_gt)
        pbar.set_postfix_str(
            f"OPEN={open_count} CLOSED={closed_count} invalid={invalid_count}"
        )

    flush_open_batch()

    metrics_output = finalize_metrics(metrics)
    if mask_hit_rate is not None:
        metrics_output["mask_hit_rate"] = mask_hit_rate
        metrics_output["mask_hit_rate_samples"] = mask_hit_rate_samples
        if mask_hit_rate_warning:
            metrics_output["mask_hit_rate_warning"] = mask_hit_rate_warning
    metrics_output["open_image_index_count"] = open_image_index_count
    metrics_output["open_image_index_retry_count"] = open_image_index_retry_count
    metrics_output["open_top_predictions"] = open_prediction_counts.most_common(10)
    metrics_output["closed_options_gold_in_candidates_rate"] = closed_option_gold_hits / max(
        1, closed_option_total
    )
    metrics_output["closed_options_gold_in_candidates_count"] = closed_option_gold_hits
    metrics_output["closed_options_total"] = closed_option_total
    metrics_output["misrouted_option_to_yesno_count"] = misrouted_option_to_yesno_count
    logger.info(
        "OPEN image-index answers: %d/%d",
        open_image_index_count,
        metrics["total_counts"]["OPEN"],
    )
    logger.info("OPEN top-10 predictions: %s", open_prediction_counts.most_common(10))
    logger.info(
        "CLOSED options gold-in-candidates rate: %.3f (%d/%d)",
        metrics_output["closed_options_gold_in_candidates_rate"],
        closed_option_gold_hits,
        closed_option_total,
    )
    logger.info(
        "CLOSED option->yes/no misroutes: %d",
        misrouted_option_to_yesno_count,
    )
    write_jsonl(args.output_dir / "gt.jsonl", gt_records)
    write_jsonl(args.output_dir / "preds.jsonl", pred_records)
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics_output, handle, indent=2)

    run_config = to_json_compatible(vars(args).copy())
    run_config["git_commit"] = get_git_commit()
    with (args.output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2)

    if args.tb:
        args.tb_logdir.mkdir(parents=True, exist_ok=True)
        writer = SafeSummaryWriter(log_dir=str(args.tb_logdir), filename_suffix=args.tb_run_name or "")
        writer.safe_add_scalar("eval/open_token_f1", metrics_output["open/token_f1"], 0)
        writer.safe_add_scalar("eval/open_exact_match", metrics_output["open/exact_match"], 0)
        writer.safe_add_scalar("eval/closed_accuracy", metrics_output["closed/accuracy"], 0)
        writer.safe_add_scalar("eval/overall_score", metrics_output["overall/score"], 0)
        writer.safe_add_scalar("eval/invalid_image", metrics_output["invalid_image"], 0)
        for subgroup, payload in metrics_output["subgroups"].items():
            writer.safe_add_text(
                f"eval/subgroup_{subgroup}",
                format_kv_table(payload["accuracy"]),
                0,
            )
        writer.safe_add_histogram("gen_open/generated_token_len", open_output_lens, 0)
        writer.safe_add_histogram("gen_open/extracted_len", open_extracted_lens, 0)
        writer.safe_add_histogram("closed/margin", closed_margins, 0)
        writer.safe_add_histogram("closed/candidate_count", closed_candidate_counts, 0)
        writer.safe_add_text("closed/mode_counts", format_kv_table(closed_mode_counts), 0)

        for idx, record in enumerate(qual_samples):
            if record.get("image_path"):
                try:
                    with Image.open(record["image_path"]) as img:
                        image = img.convert("RGB")
                except (FileNotFoundError, UnidentifiedImageError):
                    image = None
                if image is not None:
                    writer.safe_add_image(f"qual/image_{idx}", image, 0)
            writer.safe_add_text(
                f"qual/text_{idx}",
                format_kv_table(
                    {
                        "question": record.get("question"),
                        "answer_type": record.get("answer_type"),
                        "gt": record.get("gt"),
                        "pred": record.get("pred"),
                    }
                ),
                0,
            )
        writer.flush()
        writer.close()


    return metrics_output


def run_diagnose(args: argparse.Namespace) -> Dict:
    """Run diagnostic scoring on CLOSED candidates and generic answers.

    Args:
        args: Parsed evaluation args.

    Returns:
        Diagnostics dict with summary stats and optional artifacts.
    """
    set_reproducibility(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    diagnose_out_dir = args.diagnose_out_dir or (args.output_dir / "diagnose")
    diagnose_out_dir.mkdir(parents=True, exist_ok=True)
    rank_zero = is_rank_zero()
    use_tqdm = args.tqdm

    generic_set = {normalize_answer(item) for item in parse_csv_list(args.diagnose_generic_set)}
    yes_variants_expanded = parse_csv_list(args.diagnose_yes_variants)
    no_variants_expanded = parse_csv_list(args.diagnose_no_variants)
    split = "validation" if args.probe_split == "val" else "test"

    records_no_triples = load_slake_records(
        source=args.data_source,
        split=split,
        slake_root=args.slake_root,
        use_triple_context=False,
        triple_k=args.triple_k,
        max_triples=args.max_triples,
    )
    records_with_triples = load_slake_records(
        source=args.data_source,
        split=split,
        slake_root=args.slake_root,
        use_triple_context=True,
        triple_k=args.triple_k,
        max_triples=args.max_triples,
    )
    probe_base = sample_probe_records(records_no_triples, args.probe_size, args.probe_seed)
    probe_qids = [record["qid"] for record in probe_base]
    map_no_triples = {record["qid"]: record for record in records_no_triples}
    map_with_triples = {record["qid"]: record for record in records_with_triples}

    def _materialize_probe(source_map: Dict[int, Dict], triple_used: bool) -> List[Dict]:
        materialized: List[Dict] = []
        for qid in probe_qids:
            record = source_map.get(qid)
            if record is None:
                continue
            cloned = dict(record)
            cloned["triple_context_used"] = triple_used
            materialized.append(cloned)
        return materialized

    probe_no_triples = _materialize_probe(map_no_triples, False)
    probe_with_triples = _materialize_probe(map_with_triples, True)
    primary_records = probe_with_triples if args.use_triple_context else probe_no_triples

    closed_vocab: List[str] = []
    if args.closed_vocab_path:
        closed_vocab = load_closed_vocab(args.closed_vocab_path)
    elif args.allow_build_vocab:
        train_records = load_slake_records(
            source=args.data_source,
            split="train",
            slake_root=args.slake_root,
            use_triple_context=args.use_triple_context,
            triple_k=args.triple_k,
            max_triples=args.max_triples,
        )
        closed_vocab = build_closed_vocab_from_train(train_records)

    model = LlavaMedTrainable(
        model_id=args.model_id_or_ckpt,
        model_base=args.model_base,
        conv_mode=args.conv_mode,
    )
    model.model.eval()

    if args.open_stop_strings:
        open_stop_strings = [s for s in args.open_stop_strings.split(",") if s]
    else:
        open_stop_strings = resolve_stop_strings(args.conv_mode)

    def run_open_probe(records: List[Dict]) -> List[Dict]:
        open_results: List[Dict] = []
        batch_images: List[Image.Image] = []
        batch_user_texts: List[str] = []
        batch_samples: List[Dict] = []

        def flush() -> None:
            if not batch_images:
                return
            outputs, metadata = model.generate_open(
                batch_images,
                batch_user_texts,
                max_new_tokens=args.max_new_tokens_open,
                min_new_tokens=args.min_new_tokens,
                stop_strings=open_stop_strings,
                return_metadata=True,
                temperature=args.temperature,
                do_sample=args.do_sample,
            )
            for sample, output, meta in zip(batch_samples, outputs, metadata):
                raw_generation = output["raw_text"]
                extracted_pred = output["extracted_text"]
                normalized_pred = output["normalized_text"]
                normalized_gt = normalize_answer(sample["answer"])
                open_results.append(
                    {
                        "sample": sample,
                        "raw_generation": raw_generation,
                        "extracted_pred": extracted_pred,
                        "normalized_pred": normalized_pred,
                        "normalized_gt": normalized_gt,
                        "token_f1": open_token_f1(extracted_pred, sample["answer"]),
                        "generated_token_len": int(meta.get("generated_token_len", 0)),
                        "extracted_len": len(extracted_pred.split()),
                    }
                )
            batch_images.clear()
            batch_user_texts.clear()
            batch_samples.clear()

        for sample in tqdm(
            records,
            desc="Diagnose OPEN",
            unit="sample",
            dynamic_ncols=True,
            mininterval=args.tqdm_mininterval,
            leave=args.tqdm_leave,
            disable=not use_tqdm,
        ):
            if sample.get("answer_type", "").upper() != "OPEN":
                continue
            image_path = resolve_image_path(args.slake_root, sample["img_name"])
            sample["image_path"] = str(image_path)
            image = load_image(
                sample,
                args.slake_root,
                args.mask_mode,
                args.mask_threshold,
                args.mask_union_mode,
                args.mask_pad_ratio,
                use_tqdm=use_tqdm,
            )
            if image is None:
                continue
            user_text = build_user_text_open(
                sample,
                triples_str="",
                open_style=args.open_style,
            )
            batch_images.append(image)
            batch_user_texts.append(user_text)
            batch_samples.append(sample)
            if len(batch_images) >= args.batch_size:
                flush()
        flush()
        return open_results

    def summarize_open(open_results: List[Dict]) -> Dict:
        count = len(open_results)
        if not count:
            return {
                "count": 0,
                "token_f1": 0.0,
                "exact_match": 0.0,
                "generic_rate": 0.0,
                "top_preds": [],
                "avg_generated_token_len": 0.0,
                "avg_extracted_word_len": 0.0,
                "breakdowns": {},
            }
        token_f1 = sum(item["token_f1"] for item in open_results) / count
        exact_match = sum(item["normalized_pred"] == item["normalized_gt"] for item in open_results) / count
        generic_count = sum(item["normalized_pred"] in generic_set for item in open_results)
        top_preds = [
            {"pred": pred, "count": int(count)}
            for pred, count in Counter(item["normalized_pred"] for item in open_results).most_common(30)
        ]
        avg_generated_token_len = float(np.mean([item["generated_token_len"] for item in open_results]))
        avg_extracted_word_len = float(np.mean([item["extracted_len"] for item in open_results]))
        breakdowns = {}
        for subgroup in ["content_type", "base_type", "modality", "q_lang"]:
            subgroup_stats: Dict[str, Dict[str, float | int]] = {}
            for item in open_results:
                value = str(item["sample"].get(subgroup, ""))
                entry = subgroup_stats.setdefault(
                    value,
                    {"count": 0, "generic_count": 0, "token_f1_sum": 0.0},
                )
                entry["count"] += 1
                entry["generic_count"] += int(item["normalized_pred"] in generic_set)
                entry["token_f1_sum"] += item["token_f1"]
            breakdowns[f"by_{subgroup}"] = {
                key: {
                    "count": entry["count"],
                    "generic_rate": entry["generic_count"] / entry["count"] if entry["count"] else 0.0,
                    "token_f1": entry["token_f1_sum"] / entry["count"] if entry["count"] else 0.0,
                }
                for key, entry in subgroup_stats.items()
            }
        return {
            "count": count,
            "token_f1": token_f1,
            "exact_match": exact_match,
            "generic_rate": generic_count / count,
            "top_preds": top_preds,
            "avg_generated_token_len": avg_generated_token_len,
            "avg_extracted_word_len": avg_extracted_word_len,
            "breakdowns": breakdowns,
        }

    open_results_primary = run_open_probe(primary_records)
    open_results_triples_on = run_open_probe(probe_with_triples)
    open_results_triples_off = run_open_probe(probe_no_triples)

    open_summary = summarize_open(open_results_primary)
    open_summary_on = summarize_open(open_results_triples_on)
    open_summary_off = summarize_open(open_results_triples_off)

    open_examples: List[Dict] = []
    for item in open_results_primary:
        if len(open_examples) >= args.diagnose_examples:
            break
        if item["normalized_pred"] != "image":
            continue
        if item["normalized_gt"] == "image":
            continue
        sample = item["sample"]
        prompt_text = model.build_prompt(
            user_text=build_user_text_open(
                sample,
                triples_str="",
                open_style=args.open_style,
            ),
            answer_text=None,
            add_generation_prompt=True,
        )
        open_examples.append(
            {
                "qid": sample.get("qid"),
                "question": sample.get("question"),
                "base_type": sample.get("base_type"),
                "content_type": sample.get("content_type"),
                "modality": sample.get("modality"),
                "triple_context_used": sample.get("triple_context_used"),
                "prompt_text": truncate_text(prompt_text),
                "gt": sample.get("answer"),
                "pred_raw": item["raw_generation"],
                "pred_extracted": item["extracted_pred"],
                "pred_norm": item["normalized_pred"],
            }
        )

    def select_closed_candidates(sample: Dict) -> List[str]:
        """Select CLOSED candidates using routing precedence.

        Args:
            sample: Dataset record containing question/answer fields.

        Returns:
            List of candidate strings for CLOSED scoring.

        Raises:
            RuntimeError: If no candidates are available.
        """
        candidates: List[str] = []
        if args.closed_candidate_mode in {"yesno", "auto"} and is_yesno_question(
            sample["question"],
            sample["answer_type"],
            sample["answer"],
        ):
            candidates = ["yes", "no"]
        if not candidates and args.closed_candidate_mode in {"parse_options", "auto"}:
            candidates = parse_options_from_question(sample["question"])
        if not candidates and args.closed_candidate_mode in {"vocab", "auto"}:
            if not closed_vocab:
                raise RuntimeError("Closed vocab unavailable; provide --closed_vocab_path or --allow_build_vocab")
            candidates = select_topk_vocab_candidates(sample["question"], closed_vocab, args.topk_vocab_candidates)
        if not candidates:
            raise RuntimeError("No CLOSED candidates available after applying selection rules")
        return candidates

    closed_total = 0
    closed_correct = 0
    yesno_total = 0
    yesno_pred_yes = 0
    yesno_pred_no = 0
    confusion = Counter()
    yesno_margin_list: List[float] = []
    yesno_examples: List[Dict] = []
    yesno_pred_labels: List[str] = []
    yesno_variant_wins_yes = Counter()
    yesno_variant_wins_no = Counter()
    yesno_current_lengths: List[int] = []
    yesno_current_scores: List[float] = []
    yesno_expanded_lengths: List[int] = []
    yesno_expanded_scores: List[float] = []
    yesno_correct_current = 0
    yesno_correct_expanded = 0
    yesno_pred_yes_expanded = 0
    yesno_pred_no_expanded = 0

    for sample in tqdm(
        primary_records,
        desc="Diagnose CLOSED",
        unit="sample",
        dynamic_ncols=True,
        mininterval=args.tqdm_mininterval,
        leave=args.tqdm_leave,
        disable=not use_tqdm,
    ):
        if sample.get("answer_type", "").upper() != "CLOSED":
            continue
        image_path = resolve_image_path(args.slake_root, sample["img_name"])
        sample["image_path"] = str(image_path)
        image = load_image(
            sample,
            args.slake_root,
            args.mask_mode,
            args.mask_threshold,
            args.mask_union_mode,
            args.mask_pad_ratio,
            use_tqdm=use_tqdm,
        )
        if image is None:
            continue
        closed_total += 1
        normalized_gt = normalize_answer(sample["answer"])
        candidates = select_closed_candidates(sample)
        prompt = model.build_prompt(
            user_text=build_user_text_open(
                sample,
                triples_str="",
                open_style=args.open_style,
            ),
            answer_text=None,
            add_generation_prompt=True,
        )
        variant_to_base: Dict[str, str] = {}
        variant_candidates: List[str] = []
        for base in candidates:
            for variant in make_candidate_variants(base):
                if variant not in variant_to_base:
                    variant_to_base[variant] = base
                    variant_candidates.append(variant)
        if not variant_candidates:
            variant_candidates = list(candidates)
            variant_to_base = {cand: cand for cand in candidates}
        scores = model.score_closed_candidates(image, prompt, variant_candidates)
        base_scores: Dict[str, float] = {}
        for variant, score in scores.items():
            base = variant_to_base.get(variant, variant)
            if base not in base_scores or score > base_scores[base]:
                base_scores[base] = score
        if base_scores:
            best_base = max(base_scores.items(), key=lambda item: item[1])[0]
        else:
            best_base = ""
        normalized_pred = normalize_answer(best_base)
        if normalized_pred == normalized_gt:
            closed_correct += 1

        if is_yesno_question(sample["question"], sample["answer_type"], sample["answer"]):
            yesno_total += 1
            tokenizer = model.processor.tokenizer
            yes_variants_current = make_candidate_variants("yes")
            no_variants_current = make_candidate_variants("no")

            def score_yesno_set(
                yes_variants: List[str],
                no_variants: List[str],
            ) -> Dict:
                combined: List[str] = []
                seen = set()
                for variant in yes_variants + no_variants:
                    if variant not in seen:
                        combined.append(variant)
                        seen.add(variant)
                scores_local = model.score_closed_candidates(image, prompt, combined)

                def pick_best(variants: List[str]) -> tuple[str, float]:
                    best_variant = variants[0] if variants else ""
                    best_score = float("-inf")
                    for variant in variants:
                        score = scores_local.get(variant)
                        if score is None:
                            continue
                        if score > best_score:
                            best_score = score
                            best_variant = variant
                    return best_variant, float(best_score)

                best_yes_variant, best_yes_score = pick_best(yes_variants)
                best_no_variant, best_no_score = pick_best(no_variants)
                chosen_label = "yes" if best_yes_score >= best_no_score else "no"
                chosen_variant = best_yes_variant if chosen_label == "yes" else best_no_variant
                chosen_score = best_yes_score if chosen_label == "yes" else best_no_score
                token_count = int(
                    tokenizer(chosen_variant, add_special_tokens=False, return_tensors="pt")
                    .input_ids.shape[1]
                )
                margin = abs(best_yes_score - best_no_score)
                return {
                    "scores": scores_local,
                    "best_yes_variant": best_yes_variant,
                    "best_yes_score": best_yes_score,
                    "best_no_variant": best_no_variant,
                    "best_no_score": best_no_score,
                    "chosen_label": chosen_label,
                    "chosen_variant": chosen_variant,
                    "chosen_score": chosen_score,
                    "token_count": token_count,
                    "margin": margin,
                }

            current_result = score_yesno_set(yes_variants_current, no_variants_current)
            expanded_result = score_yesno_set(yes_variants_expanded, no_variants_expanded)

            pred_label = current_result["chosen_label"]
            yesno_pred_labels.append(pred_label)
            yesno_pred_yes += int(pred_label == "yes")
            yesno_pred_no += int(pred_label == "no")
            if normalized_gt == "yes":
                confusion["gt_yes_pred_yes"] += int(pred_label == "yes")
                confusion["gt_yes_pred_no"] += int(pred_label == "no")
            elif normalized_gt == "no":
                confusion["gt_no_pred_yes"] += int(pred_label == "yes")
                confusion["gt_no_pred_no"] += int(pred_label == "no")
            if normalized_gt in {"yes", "no"} and pred_label == normalized_gt:
                yesno_correct_current += 1
            if normalized_gt in {"yes", "no"} and expanded_result["chosen_label"] == normalized_gt:
                yesno_correct_expanded += 1
            yesno_pred_yes_expanded += int(expanded_result["chosen_label"] == "yes")
            yesno_pred_no_expanded += int(expanded_result["chosen_label"] == "no")

            yesno_margin_list.append(current_result["margin"])
            yesno_variant_wins_yes[current_result["best_yes_variant"]] += 1
            yesno_variant_wins_no[current_result["best_no_variant"]] += 1

            for variant, score in current_result["scores"].items():
                token_len = int(
                    tokenizer(variant, add_special_tokens=False, return_tensors="pt")
                    .input_ids.shape[1]
                )
                yesno_current_lengths.append(token_len)
                yesno_current_scores.append(float(score))
            for variant, score in expanded_result["scores"].items():
                token_len = int(
                    tokenizer(variant, add_special_tokens=False, return_tensors="pt")
                    .input_ids.shape[1]
                )
                yesno_expanded_lengths.append(token_len)
                yesno_expanded_scores.append(float(score))

            if len(yesno_examples) < args.diagnose_examples and pred_label != normalized_gt:
                yesno_examples.append(
                    {
                        "qid": sample.get("qid"),
                        "question": sample.get("question"),
                        "base_type": sample.get("base_type"),
                        "content_type": sample.get("content_type"),
                        "modality": sample.get("modality"),
                        "triple_context_used": sample.get("triple_context_used"),
                        "prompt_text": truncate_text(prompt),
                        "gt": sample.get("answer"),
                        "best_yes_variant": current_result["best_yes_variant"],
                        "best_yes_score": current_result["best_yes_score"],
                        "best_no_variant": current_result["best_no_variant"],
                        "best_no_score": current_result["best_no_score"],
                        "margin": current_result["margin"],
                        "chosen_label": pred_label,
                        "token_count": current_result["token_count"],
                    }
                )

    closed_accuracy = closed_correct / closed_total if closed_total else 0.0
    yesno_pred_yes_rate = yesno_pred_yes / yesno_total if yesno_total else 0.0
    yesno_pred_no_rate = yesno_pred_no / yesno_total if yesno_total else 0.0
    yesno_accuracy_yes = (
        confusion["gt_yes_pred_yes"] / max(1, confusion["gt_yes_pred_yes"] + confusion["gt_yes_pred_no"])
    )
    yesno_accuracy_no = (
        confusion["gt_no_pred_no"] / max(1, confusion["gt_no_pred_no"] + confusion["gt_no_pred_yes"])
    )
    yesno_margin_stats = {
        "mean": float(np.mean(yesno_margin_list)) if yesno_margin_list else 0.0,
        "median": float(np.median(yesno_margin_list)) if yesno_margin_list else 0.0,
        "p90": float(np.percentile(yesno_margin_list, 90)) if yesno_margin_list else 0.0,
    }
    majority_label = "yes" if yesno_pred_yes >= yesno_pred_no else "no"
    tiny_margin_count = sum(
        margin < 0.05 and pred == majority_label
        for margin, pred in zip(yesno_margin_list, yesno_pred_labels)
    )
    tiny_margin_rate = tiny_margin_count / yesno_total if yesno_total else 0.0
    def _format_top_variant(counter: Counter, total: int) -> Dict[str, float | str | int]:
        if not counter:
            return {"variant": "", "count": 0, "rate": 0.0}
        variant, count = counter.most_common(1)[0]
        return {"variant": variant, "count": int(count), "rate": count / max(1, total)}

    yesno_variant_bias = {
        "yes": _format_top_variant(yesno_variant_wins_yes, yesno_total),
        "no": _format_top_variant(yesno_variant_wins_no, yesno_total),
    }

    def _corrcoef(lengths: List[int], scores: List[float]) -> float:
        if len(lengths) < 2:
            return 0.0
        corr = float(np.corrcoef(lengths, scores)[0, 1])
        if not np.isfinite(corr):
            return 0.0
        return corr

    length_score_corr_current = _corrcoef(yesno_current_lengths, yesno_current_scores)
    length_score_corr_expanded = _corrcoef(yesno_expanded_lengths, yesno_expanded_scores)

    open_ablation = {
        "triples_on": {
            "generic_rate": open_summary_on["generic_rate"],
            "token_f1": open_summary_on["token_f1"],
        },
        "triples_off_or_kvqa_only": {
            "generic_rate": open_summary_off["generic_rate"],
            "token_f1": open_summary_off["token_f1"],
        },
        "delta": {
            "generic_rate": open_summary_on["generic_rate"] - open_summary_off["generic_rate"],
            "token_f1": open_summary_on["token_f1"] - open_summary_off["token_f1"],
        },
    }

    closed_yesno_current = {
        "accuracy": yesno_correct_current / yesno_total if yesno_total else 0.0,
        "pred_yes": yesno_pred_yes,
        "pred_no": yesno_pred_no,
    }
    closed_yesno_expanded = {
        "accuracy": yesno_correct_expanded / yesno_total if yesno_total else 0.0,
        "pred_yes": yesno_pred_yes_expanded,
        "pred_no": yesno_pred_no_expanded,
    }
    closed_yesno_delta = {
        "accuracy": closed_yesno_expanded["accuracy"] - closed_yesno_current["accuracy"],
        "pred_yes": closed_yesno_expanded["pred_yes"] - closed_yesno_current["pred_yes"],
        "pred_no": closed_yesno_expanded["pred_no"] - closed_yesno_current["pred_no"],
    }

    diagnostics = {
        "run_info": {"split": split, "probe_size": args.probe_size, "seed": args.probe_seed},
        "open": {
            "count": open_summary["count"],
            "token_f1": open_summary["token_f1"],
            "exact_match": open_summary["exact_match"],
            "generic_rate": open_summary["generic_rate"],
            "top_preds": open_summary["top_preds"],
            "avg_generated_token_len": open_summary["avg_generated_token_len"],
            "avg_extracted_word_len": open_summary["avg_extracted_word_len"],
            "breakdowns": open_summary["breakdowns"],
            "ablation": open_ablation,
        },
        "closed": {
            "count": closed_total,
            "accuracy": closed_accuracy,
            "yesno": {
                "count": yesno_total,
                "pred_yes": yesno_pred_yes,
                "pred_no": yesno_pred_no,
                "pred_yes_rate": yesno_pred_yes_rate,
                "pred_no_rate": yesno_pred_no_rate,
                "confusion": {
                    "gt_yes_pred_yes": confusion["gt_yes_pred_yes"],
                    "gt_yes_pred_no": confusion["gt_yes_pred_no"],
                    "gt_no_pred_yes": confusion["gt_no_pred_yes"],
                    "gt_no_pred_no": confusion["gt_no_pred_no"],
                },
                "accuracy_yes_subset": yesno_accuracy_yes,
                "accuracy_no_subset": yesno_accuracy_no,
                "margin_stats": yesno_margin_stats,
                "tiny_margin_majority_rate": tiny_margin_rate,
                "variant_winner_top1": yesno_variant_bias,
                "length_score_corr": {
                    "current": length_score_corr_current,
                    "variants_expanded": length_score_corr_expanded,
                },
                "ablation": {
                    "current": closed_yesno_current,
                    "variants_expanded": closed_yesno_expanded,
                    "delta": closed_yesno_delta,
                },
            },
        },
    }

    if rank_zero:
        with (diagnose_out_dir / "diagnostics.json").open("w", encoding="utf-8") as handle:
            json.dump(diagnostics, handle, indent=2)

        if open_summary["top_preds"]:
            with (diagnose_out_dir / "open_top_preds.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["pred", "count"])
                writer.writeheader()
                for row in open_summary["top_preds"]:
                    writer.writerow(row)

        with (diagnose_out_dir / "closed_yesno_report.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
            writer.writeheader()
            writer.writerow({"metric": "count", "value": yesno_total})
            writer.writerow({"metric": "pred_yes", "value": yesno_pred_yes})
            writer.writerow({"metric": "pred_no", "value": yesno_pred_no})
            writer.writerow({"metric": "pred_yes_rate", "value": yesno_pred_yes_rate})
            writer.writerow({"metric": "pred_no_rate", "value": yesno_pred_no_rate})
            writer.writerow({"metric": "gt_yes_pred_yes", "value": confusion["gt_yes_pred_yes"]})
            writer.writerow({"metric": "gt_yes_pred_no", "value": confusion["gt_yes_pred_no"]})
            writer.writerow({"metric": "gt_no_pred_yes", "value": confusion["gt_no_pred_yes"]})
            writer.writerow({"metric": "gt_no_pred_no", "value": confusion["gt_no_pred_no"]})
            writer.writerow({"metric": "accuracy_yes_subset", "value": yesno_accuracy_yes})
            writer.writerow({"metric": "accuracy_no_subset", "value": yesno_accuracy_no})
            writer.writerow({"metric": "margin_mean", "value": yesno_margin_stats["mean"]})
            writer.writerow({"metric": "margin_median", "value": yesno_margin_stats["median"]})
            writer.writerow({"metric": "margin_p90", "value": yesno_margin_stats["p90"]})

        write_jsonl(diagnose_out_dir / "examples_open_image.jsonl", open_examples)
        write_jsonl(diagnose_out_dir / "examples_closed_yesno.jsonl", yesno_examples)

    return diagnostics


def main() -> None:
    """Entry point for evaluation."""
    logging.basicConfig(level=logging.INFO)
    args = build_parser().parse_args()
    args.tqdm = True
    if args.mode == "diagnose":
        run_diagnose(args)
    else:
        run_evaluation(args)


if __name__ == "__main__":
    main()
