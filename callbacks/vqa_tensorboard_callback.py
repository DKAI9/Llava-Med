"""TensorBoard logging callback for VQA training."""
from __future__ import annotations

import json
import logging
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from utils.mask_preprocess import apply_mask, load_mask
from utils.prompting import build_triples_context, build_user_text_open, resolve_stop_strings
from utils.tensorboard_utils import SafeSummaryWriter, format_kv_table
from utils.text_norm import normalize_answer, open_token_f1

logger = logging.getLogger(__name__)


def load_image(
    image_path: str,
    sample: Dict,
    slake_root: Path,
    mask_mode: str,
    mask_threshold: int,
    mask_union_mode: str,
    mask_pad_ratio: float,
) -> Tuple[Image.Image, bool]:
    """Load an image and apply masks if configured.

    Args:
        image_path: Path to the image file.
        sample: Sample record for mask resolution.
        slake_root: SLAKE dataset root directory.
        mask_mode: Masking mode (``none``, ``masked``, ``crop``).
        mask_threshold: Threshold for binary mask conversion.
        mask_union_mode: Mask union strategy.
        mask_pad_ratio: Padding ratio for crop masks.

    Returns:
        Tuple of (image, used_placeholder) where placeholder indicates fallback.
    """
    try:
        with Image.open(image_path) as img:
            image = img.convert("RGB")
    except (FileNotFoundError, UnidentifiedImageError) as exc:
        logger.warning("Invalid image %s: %s", image_path, exc)
        return Image.new("RGB", (224, 224), color=0), True
    if mask_mode != "none":
        mask = load_mask(sample, slake_root, mask_threshold, mask_union_mode)
        image = apply_mask(image, mask, mask_mode, mask_pad_ratio)
    return image, False


class VQATensorBoardCallback(TrainerCallback):
    """TensorBoard callback for VQA training/evaluation metrics."""
    def __init__(
        self,
        writer: SafeSummaryWriter,
        trainer,
        probe_records: List[Dict],
        closed_vocab: Optional[Sequence[str]],
        slake_root: Optional[Path] = None,
        tb_log_images_every: int = 500,
        tb_log_text_every: int = 200,
        tb_eval_generate_size: int = 128,
        tb_hist_every: int = 200,
        open_min_new_tokens: int = 1,
        mask_mode: str = "none",
        mask_pad_ratio: float = 0.1,
        mask_threshold: int = 1,
        mask_union_mode: str = "prefer_disease_for_abnormality",
        triples_mode: str = "real_only",
        open_style: str = "short",
        closed_style: str = "minimal",
        closed_yesno_variants: bool = True,
    ) -> None:
        """Initialize the callback.

        Args:
            writer: SafeSummaryWriter instance.
            trainer: HuggingFace Trainer.
            probe_records: Fixed probe samples for periodic eval.
            closed_vocab: CLOSED vocabulary for candidate scoring.
            slake_root: Optional root for image/mask files.
            tb_log_images_every: Steps between image logs.
            tb_log_text_every: Steps between text logs.
            tb_eval_generate_size: Number of probe samples to evaluate.
            tb_hist_every: Steps between histogram logs.
            open_min_new_tokens: Minimum tokens to generate for OPEN eval.
            mask_mode: Masking mode for images.
            mask_pad_ratio: Padding ratio for crop masks.
            mask_threshold: Threshold for binary masks.
            mask_union_mode: Mask union strategy.
            triples_mode: Triples mode for prompt context.
            open_style: Prompt style for OPEN.
            closed_style: Prompt style for CLOSED.
            closed_yesno_variants: Whether to use yes/no variants in scoring.
        """
        self.writer = writer
        self.trainer = trainer
        self.probe_records = probe_records
        self.closed_vocab = list(closed_vocab) if closed_vocab is not None else []
        self.slake_root = slake_root
        self.tb_log_images_every = tb_log_images_every
        self.tb_log_text_every = tb_log_text_every
        self.tb_eval_generate_size = tb_eval_generate_size
        self.tb_hist_every = tb_hist_every
        self.open_min_new_tokens = int(open_min_new_tokens)
        self.mask_mode = mask_mode
        self.mask_pad_ratio = float(mask_pad_ratio)
        self.mask_threshold = int(mask_threshold)
        self.mask_union_mode = mask_union_mode
        self.triples_mode = triples_mode
        self.open_style = open_style
        self.closed_style = closed_style
        self.closed_yesno_variants = closed_yesno_variants
        self._tokens_since_log = 0
        self._last_log_time = time.time()
        self._last_grad_norm: Optional[float] = None

    def _is_main_process(self, state: TrainerState) -> bool:
        """Return True if the current process is rank 0.

        Args:
            state: Trainer state containing distributed rank info.

        Returns:
            True if this is the main process.
        """
        return state.is_world_process_zero

    def _log_dataset_stats(self, records: Iterable[Dict], step: int) -> None:
        """Log dataset distribution stats to TensorBoard.

        Args:
            records: Iterable of dataset records.
            step: Global step for logging.
        """
        counts = defaultdict(Counter)
        if hasattr(records, "__len__") and hasattr(records, "__getitem__") and not isinstance(records, list):
            iterable = (records[idx] for idx in range(len(records)))
        else:
            iterable = records
        for record in iterable:
            counts["answer_type"][record.get("answer_type", "")] += 1
            counts["q_lang"][record.get("q_lang", "")] += 1
            counts["modality"][record.get("modality", "")] += 1
            counts["content_type"][record.get("content_type", "")] += 1
            counts["base_type"][record.get("base_type", "")] += 1

        for group, counter in counts.items():
            for key, value in counter.items():
                self.writer.safe_add_scalar(f"data/{group}/{key}", value, step)
            self.writer.safe_add_text(
                f"data/{group}_table",
                format_kv_table(dict(counter)),
                step,
            )

    def _log_trainable_params(self, step: int) -> None:
        """Log trainable parameter counts if the model exposes them.

        Args:
            step: Global step for logging.
        """
        if not hasattr(self.trainer.model, "count_trainable_params"):
            return
        counts = self.trainer.model.count_trainable_params()
        for key, value in counts.items():
            self.writer.safe_add_scalar(f"train/params/{key}", value, step)

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        """Log run configuration and dataset stats at the start of training.

        Args:
            args: Training arguments.
            state: Trainer state.
            control: Trainer control flags.
            **kwargs: Additional callback arguments.
        """
        if not self._is_main_process(state):
            return
        config_text = json.dumps(vars(args), indent=2, default=str)
        self.writer.safe_add_text("run/config", config_text, state.global_step)
        self._log_dataset_stats(self.trainer.train_dataset, state.global_step)
        self._log_trainable_params(state.global_step)

    def on_pre_optimizer_step(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        """Capture gradient norm before the optimizer step.

        Args:
            args: Training arguments.
            state: Trainer state.
            control: Trainer control flags.
            **kwargs: Additional callback arguments.
        """
        if not self._is_main_process(state):
            return
        params = [p for p in self.trainer.model.parameters() if p.grad is not None]
        if not params:
            self._last_grad_norm = None
            return
        total = torch.zeros((), device=params[0].device)
        for param in params:
            total += param.grad.detach().float().norm(2) ** 2
        self._last_grad_norm = float(torch.sqrt(total).item())

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        """Accumulate supervised token count for throughput metrics.

        Args:
            args: Training arguments.
            state: Trainer state.
            control: Trainer control flags.
            **kwargs: Additional callback arguments.
        """
        if not self._is_main_process(state):
            return
        batch = getattr(self.trainer, "last_batch", None)
        if not batch:
            return
        answer_len = batch.get("answer_len")
        if isinstance(answer_len, torch.Tensor):
            self._tokens_since_log += int(answer_len.sum().item())

    def _log_batch_stats(self, batch: Dict, step: int) -> None:
        """Log per-batch statistics (lengths, losses, routing ratios).

        Args:
            batch: Batch dict from the collator/trainer.
            step: Global step for logging.
        """
        prompt_len = batch.get("prompt_len")
        answer_len = batch.get("answer_len")
        prompt_truncated = batch.get("prompt_truncated")
        answer_truncated = batch.get("answer_truncated")
        placeholder = batch.get("used_placeholder_image")
        answer_types = batch.get("answer_type", [])
        sft_loss = batch.get("sft_loss")
        mc_loss = batch.get("mc_loss")
        mc_valid_count = batch.get("mc_valid_count")
        mc_closed_count = batch.get("mc_closed_count")
        mc_route_counts = batch.get("mc_route_counts")
        mc_acc_yesno = batch.get("mc_acc_yesno")
        mc_acc_options = batch.get("mc_acc_options")
        mc_acc_vocab = batch.get("mc_acc_vocab")
        mc_yesno_pred_yes = batch.get("mc_yesno_pred_yes")
        mc_yesno_pred_no = batch.get("mc_yesno_pred_no")

        def _mean_max(tensor: torch.Tensor) -> Tuple[float, float]:
            return float(tensor.float().mean().item()), float(tensor.float().max().item())

        if isinstance(prompt_len, torch.Tensor):
            mean_val, max_val = _mean_max(prompt_len)
            self.writer.safe_add_scalar("data/prompt_len_mean", mean_val, step)
            self.writer.safe_add_scalar("data/prompt_len_max", max_val, step)
        if isinstance(answer_len, torch.Tensor):
            mean_val, max_val = _mean_max(answer_len)
            self.writer.safe_add_scalar("data/answer_len_mean", mean_val, step)
            self.writer.safe_add_scalar("data/answer_len_max", max_val, step)
            self.writer.safe_add_scalar("train/supervised_tokens", float(answer_len.sum().item()), step)
        if isinstance(prompt_truncated, torch.Tensor):
            self.writer.safe_add_scalar(
                "data/prompt_truncated_pct", float(prompt_truncated.float().mean().item()), step
            )
        if isinstance(answer_truncated, torch.Tensor):
            self.writer.safe_add_scalar(
                "data/answer_truncated_pct", float(answer_truncated.float().mean().item()), step
            )
        if isinstance(placeholder, torch.Tensor):
            self.writer.safe_add_scalar(
                "data/placeholder_image_pct", float(placeholder.float().mean().item()), step
            )
        if answer_types:
            total = len(answer_types)
            open_count = sum(1 for item in answer_types if str(item).upper() == "OPEN")
            closed_count = total - open_count
            if total:
                self.writer.safe_add_scalar("data/open_ratio", open_count / total, step)
                self.writer.safe_add_scalar("data/closed_ratio", closed_count / total, step)
        if isinstance(sft_loss, float):
            self.writer.safe_add_scalar("train/sft_loss", sft_loss, step)
        if isinstance(mc_loss, float):
            self.writer.safe_add_scalar("train/mc_loss", mc_loss, step)
        if isinstance(mc_valid_count, int) and isinstance(mc_closed_count, int):
            self.writer.safe_add_scalar(
                "train/mc_valid_ratio",
                mc_valid_count / max(mc_closed_count, 1),
                step,
            )
        if isinstance(mc_route_counts, dict):
            self.writer.safe_add_text("train/mc_route_counts", format_kv_table(mc_route_counts), step)
        if isinstance(mc_acc_yesno, float):
            self.writer.safe_add_scalar("train/mc_acc_yesno", mc_acc_yesno, step)
        if isinstance(mc_acc_options, float):
            self.writer.safe_add_scalar("train/mc_acc_options", mc_acc_options, step)
        if isinstance(mc_acc_vocab, float):
            self.writer.safe_add_scalar("train/mc_acc_vocab", mc_acc_vocab, step)
        if isinstance(mc_yesno_pred_yes, int):
            self.writer.safe_add_scalar("train/mc_yesno_pred_yes", mc_yesno_pred_yes, step)
        if isinstance(mc_yesno_pred_no, int):
            self.writer.safe_add_scalar("train/mc_yesno_pred_no", mc_yesno_pred_no, step)

    def _log_histograms(self, batch: Dict, step: int) -> None:
        """Log histogram summaries for prompt/answer lengths.

        Args:
            batch: Batch dict from the collator/trainer.
            step: Global step for logging.
        """
        prompt_len = batch.get("prompt_len")
        answer_len = batch.get("answer_len")
        if isinstance(prompt_len, torch.Tensor):
            self.writer.safe_add_histogram("data/prompt_len_hist", prompt_len, step)
        if isinstance(answer_len, torch.Tensor):
            self.writer.safe_add_histogram("data/answer_len_hist", answer_len, step)
            self.writer.safe_add_histogram("data/supervised_tokens_hist", answer_len, step)

    def _log_throughput(self, step: int) -> None:
        """Log tokens-per-second throughput based on recent steps.

        Args:
            step: Global step for logging.
        """
        now = time.time()
        elapsed = now - self._last_log_time
        if elapsed > 0:
            tokens_per_sec = self._tokens_since_log / elapsed
            self.writer.safe_add_scalar("train/tokens_per_sec", tokens_per_sec, step)
        self._tokens_since_log = 0
        self._last_log_time = now

    def _log_gpu_metrics(self, step: int) -> None:
        """Log GPU memory usage if CUDA is available.

        Args:
            step: Global step for logging.
        """
        if not torch.cuda.is_available():
            return
        device = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(device) / (1024**2)
        reserved = torch.cuda.memory_reserved(device) / (1024**2)
        self.writer.safe_add_scalar("gpu/memory_allocated_mb", allocated, step)
        self.writer.safe_add_scalar("gpu/memory_reserved_mb", reserved, step)

    def _log_param_norms(self, step: int) -> None:
        """Log parameter norms for selected model components.

        Args:
            step: Global step for logging.
        """
        if not hasattr(self.trainer.model, "get_param_norms"):
            return
        norms = self.trainer.model.get_param_norms(["mm_projector", "lora"])
        for key, value in norms.items():
            self.writer.safe_add_scalar(f"train/param_norm/{key}", value, step)

    def _run_probe_eval(self, step: int, log_images: bool, log_text: bool) -> None:
        """Run a lightweight probe evaluation for qualitative monitoring.

        Args:
            step: Global step for logging.
            log_images: Whether to log images to TensorBoard.
            log_text: Whether to log text samples to TensorBoard.
        """
        if not self.probe_records:
            return
        model = self.trainer.model
        was_training = model.model.training
        model.model.eval()
        probe = self.probe_records[: self.tb_eval_generate_size]

        open_samples = [s for s in probe if str(s.get("answer_type", "")).upper() == "OPEN"]
        closed_samples = [s for s in probe if str(s.get("answer_type", "")).upper() == "CLOSED"]

        open_outputs: List[Dict[str, str]] = []
        open_metadata: List[Dict[str, int | str | None]] = []
        open_gt: List[str] = []
        open_user_texts: List[str] = []
        open_images: List[Image.Image] = []
        stop_strings = resolve_stop_strings(model.conv_mode)

        for sample in open_samples:
            slake_root = self.slake_root or Path(sample["image_path"]).parents[2]
            image, _ = load_image(
                sample["image_path"],
                sample,
                slake_root,
                self.mask_mode,
                self.mask_threshold,
                self.mask_union_mode,
                self.mask_pad_ratio,
            )
            open_images.append(image)
            open_user_texts.append(
                build_user_text_open(
                    sample,
                    triples_str=build_triples_context(sample, self.triples_mode),
                    open_style=self.open_style,
                )
            )
            open_gt.append(sample.get("answer", ""))

        batch_size = max(1, min(4, len(open_images)))
        for idx in range(0, len(open_images), batch_size):
            batch_images = open_images[idx : idx + batch_size]
            batch_user_texts = open_user_texts[idx : idx + batch_size]
            outputs, metadata = model.generate_open(
                batch_images,
                batch_user_texts,
                max_new_tokens=16,
                stop_strings=stop_strings,
                return_metadata=True,
                min_new_tokens=self.open_min_new_tokens,
            )
            open_outputs.extend(outputs)
            open_metadata.extend(metadata)

        open_lengths = [meta["generated_token_len"] for meta in open_metadata]
        stop_reasons = Counter(meta["stopped_by"] for meta in open_metadata)
        open_correct = 0
        open_f1_total = 0.0
        for output, gt in zip(open_outputs, open_gt):
            extracted = output["extracted_text"]
            normalized = output["normalized_text"]
            if normalized == normalize_answer(gt):
                open_correct += 1
            open_f1_total += open_token_f1(extracted, gt)

        if open_outputs:
            self.writer.safe_add_histogram("gen_open/generated_token_len", open_lengths, step)
            self.writer.safe_add_text("gen_open/stop_reason_counts", format_kv_table(stop_reasons), step)
            self.writer.safe_add_scalar(
                "gen_open/em_probe",
                open_correct / max(1, len(open_outputs)),
                step,
            )
            self.writer.safe_add_scalar(
                "gen_open/token_f1_probe",
                open_f1_total / max(1, len(open_outputs)),
                step,
            )

        closed_margins: List[float] = []
        closed_modes = Counter()
        closed_candidate_counts: List[int] = []
        closed_correct = 0
        closed_text_blocks: List[str] = []
        cfg = {
            "closed_yesno_variants": self.closed_yesno_variants,
            "closed_use_vocab_fallback": True,
            "closed_style": self.closed_style,
            "triples_mode": self.triples_mode,
        }
        for sample in closed_samples:
            slake_root = self.slake_root or Path(sample["image_path"]).parents[2]
            image, _ = load_image(
                sample["image_path"],
                sample,
                slake_root,
                self.mask_mode,
                self.mask_threshold,
                self.mask_union_mode,
                self.mask_pad_ratio,
            )
            pred, debug = model.score_closed(sample, image, self.closed_vocab, cfg)
            closed_candidate_counts.append(int(debug.get("candidate_count", 0)))
            closed_modes[debug.get("route", "vocab")] += 1
            closed_margins.append(float(debug.get("margin", 0.0)))
            if normalize_answer(pred) == normalize_answer(sample.get("answer", "")):
                closed_correct += 1
            if log_text and len(closed_text_blocks) < 3:
                closed_text_blocks.append(
                    "\n".join(
                        [
                            f"Q: {sample.get('question')}",
                            f"GT: {sample.get('answer')}",
                            f"PRED: {pred}",
                            f"ROUTE: {debug.get('route')}",
                        ]
                    )
                )

        if closed_samples:
            self.writer.safe_add_histogram("closed/margin", closed_margins, step)
            self.writer.safe_add_histogram("closed/candidate_count", closed_candidate_counts, step)
            self.writer.safe_add_text("closed/mode_counts", format_kv_table(closed_modes), step)
            self.writer.safe_add_scalar(
                "closed/em_probe",
                closed_correct / max(1, len(closed_samples)),
                step,
            )

        if log_text:
            for idx, (output, gt, question) in enumerate(zip(open_outputs[:3], open_gt, open_user_texts)):
                text_block = "\n".join(
                    [
                        f"Q: {question}",
                        f"GT: {gt}",
                        f"PRED: {output['raw_text']}",
                        f"EXTRACTED: {output['extracted_text']}",
                        f"NORM: {output['normalized_text']}",
                    ]
                )
                self.writer.safe_add_text(f"qual/open_example_{idx}", text_block, step)
            for idx, block in enumerate(closed_text_blocks):
                self.writer.safe_add_text(f"qual/closed_example_{idx}", block, step)

        if log_images:
            for idx, image in enumerate(open_images[:3]):
                self.writer.safe_add_image(f"qual/image_open_{idx}", image, step)

        if was_training:
            model.model.train()

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> None:
        """Log metrics and periodic probe outputs during training.

        Args:
            args: Training arguments.
            state: Trainer state.
            control: Trainer control flags.
            logs: Optional log dict from Trainer.
            **kwargs: Additional callback arguments.
        """
        if not self._is_main_process(state):
            return
        if logs:
            for key, value in logs.items():
                if key in {"loss", "learning_rate"}:
                    self.writer.safe_add_scalar(f"train/{key}", value, state.global_step)
        batch = getattr(self.trainer, "last_batch", None)
        if batch:
            self._log_batch_stats(batch, state.global_step)
            if self.tb_hist_every and state.global_step % self.tb_hist_every == 0:
                self._log_histograms(batch, state.global_step)

        if self._last_grad_norm is not None:
            self.writer.safe_add_scalar("train/grad_norm", self._last_grad_norm, state.global_step)
        self._log_param_norms(state.global_step)
        self._log_gpu_metrics(state.global_step)
        self._log_throughput(state.global_step)

        log_images = self.tb_log_images_every and state.global_step % self.tb_log_images_every == 0
        log_text = self.tb_log_text_every and state.global_step % self.tb_log_text_every == 0
        if log_images or log_text:
            self._run_probe_eval(state.global_step, log_images=log_images, log_text=log_text)
        self.writer.flush()

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        """Close the TensorBoard writer at training end.

        Args:
            args: Training arguments.
            state: Trainer state.
            control: Trainer control flags.
            **kwargs: Additional callback arguments.
        """
        if not self._is_main_process(state):
            return
        self.writer.close()
