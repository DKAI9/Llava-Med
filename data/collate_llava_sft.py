"""Data collator for LLaVA-Med SFT training on SLAKE."""
from __future__ import annotations

from typing import Dict, List

import torch
from PIL import Image, UnidentifiedImageError
from transformers import PreTrainedTokenizerBase

from utils.closed_router import parse_explicit_options
from utils.prompting import (
    build_prompt_pair,
    build_triples_context,
    build_user_text_closed,
    build_user_text_open,
)


def build_prompt(
    user_text: str,
    answer: str,
    conv_mode: str,
    mm_use_im_start_end: bool,
) -> tuple[str, str]:
    """Build prompts with and without the answer for supervised training.

    Args:
        user_text: User message containing the question (and options/context).
        answer: Ground-truth answer text to append.
        conv_mode: Conversation template key for LLaVA.
        mm_use_im_start_end: Whether to include <im_start>/<im_end> tokens.

    Returns:
        Tuple of ``(prompt_without_answer, prompt_with_answer)`` strings.
    """
    return build_prompt_pair(
        conv_mode=conv_mode,
        user_text=user_text,
        answer_text=answer,
        mm_use_im_start_end=mm_use_im_start_end,
    )


class LlavaSFTCollator:
    """Collator that loads images and builds masked labels."""

    def __init__(
        self,
        processor,
        max_prompt_len: int = 512,
        max_answer_len: int = 64,
        conv_mode: str = "mistral_instruct",
        mm_use_im_start_end: bool = False,
        open_style: str = "short",
        closed_style: str = "minimal",
        triples_mode: str = "real_only",
    ) -> None:
        """Initialize the collator.

        Args:
            processor: LLaVA processor providing tokenizer and image processor.
            max_prompt_len: Maximum prompt token length (prompt-only).
            max_answer_len: Maximum answer token length (answer-only).
            conv_mode: Conversation template key for LLaVA.
            mm_use_im_start_end: Whether to include <im_start>/<im_end> tokens.
            open_style: Prompt style for OPEN answers.
            closed_style: Prompt style for CLOSED answers.
            triples_mode: Whether to include KG triples in the prompt.
        """
        self.processor = processor
        self.tokenizer: PreTrainedTokenizerBase = processor.tokenizer
        self.max_prompt_len = max_prompt_len
        self.max_answer_len = max_answer_len
        self.conv_mode = conv_mode
        self.mm_use_im_start_end = mm_use_im_start_end
        self.open_style = open_style
        self.closed_style = closed_style
        self.triples_mode = triples_mode

    def _tokenize_text(self, text: str) -> List[int]:
        """Tokenize text and truncate to the prompt limit.

        Args:
            text: Input string to tokenize.

        Returns:
            List of token ids truncated to ``max_prompt_len``.
        """
        if getattr(self.processor, "is_llava_official", False):
            from llava.constants import IMAGE_TOKEN_INDEX
            from llava.mm_utils import tokenizer_image_token

            ids = tokenizer_image_token(
                text,
                self.tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors=None,
            )
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            return ids[: self.max_prompt_len]
        tokens = self.tokenizer(text, add_special_tokens=False)
        return tokens["input_ids"][: self.max_prompt_len]

    def _tokenize_text_full(self, text: str) -> List[int]:
        """Tokenize text without truncation.

        Args:
            text: Input string to tokenize.

        Returns:
            Full list of token ids.
        """
        if getattr(self.processor, "is_llava_official", False):
            from llava.constants import IMAGE_TOKEN_INDEX
            from llava.mm_utils import tokenizer_image_token

            ids = tokenizer_image_token(
                text,
                self.tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors=None,
            )
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            return ids
        tokens = self.tokenizer(text, add_special_tokens=False)
        return tokens["input_ids"]

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor | List]:
        """Collate a batch into model-ready tensors.

        Args:
            batch: List of SLAKE samples containing question/answer/image fields.

        Returns:
            Dict with padded ``input_ids``, ``attention_mask``, ``labels``,
            ``pixel_values``, and metadata fields (qid, answer_type, etc.).
        """
        images = []
        labels_list = []
        input_ids_list = []
        attention_masks = []
        qids = []
        answer_types = []
        questions = []
        answers = []
        content_types = []
        base_types = []
        modalities = []
        triples = []
        prompt_lens = []
        answer_lens = []
        prompt_truncated = []
        answer_truncated = []
        placeholder_images = []

        for sample in batch:
            qids.append(sample["qid"])
            answer_types.append(sample["answer_type"])
            questions.append(sample.get("question", ""))
            answers.append(sample.get("answer", ""))
            content_types.append(sample.get("content_type", ""))
            base_types.append(sample.get("base_type", ""))
            modalities.append(sample.get("modality", ""))
            triples.append(sample.get("triple", []))
            triples_str = build_triples_context(sample, self.triples_mode)
            if str(sample.get("answer_type", "")).upper() == "CLOSED":
                options = parse_explicit_options(sample.get("question", ""))
                user_text = build_user_text_closed(
                    sample,
                    options=options,
                    triples_str=triples_str,
                    closed_style=self.closed_style,
                )
                answer_text = str(sample.get("answer", ""))
                normalized = answer_text.strip().lower()
                if normalized in {"yes", "no"}:
                    answer_text = "Yes" if normalized == "yes" else "No"
            else:
                user_text = build_user_text_open(
                    sample,
                    triples_str=triples_str,
                    open_style=self.open_style,
                )
                answer_text = str(sample.get("answer", ""))
            prompt_no_answer, prompt_with_answer = build_prompt(
                user_text,
                answer_text,
                conv_mode=self.conv_mode,
                mm_use_im_start_end=self.mm_use_im_start_end,
            )

            prompt_ids_full = self._tokenize_text_full(prompt_no_answer)
            full_ids = self._tokenize_text_full(prompt_with_answer)
            # Track truncation so training metrics can report prompt/answer loss coverage.
            prompt_truncated.append(len(prompt_ids_full) > self.max_prompt_len)
            prompt_ids = prompt_ids_full[: self.max_prompt_len]
            answer_ids_full = full_ids[len(prompt_ids_full) :]
            answer_truncated.append(len(answer_ids_full) > self.max_answer_len)
            answer_ids = answer_ids_full[: self.max_answer_len]

            input_ids = prompt_ids + answer_ids
            # Mask prompt tokens with -100 to exclude from supervised loss.
            labels = [-100] * len(prompt_ids) + answer_ids

            prompt_lens.append(len(prompt_ids))
            answer_lens.append(len(answer_ids))

            input_ids_list.append(torch.tensor(input_ids, dtype=torch.long))
            attention_masks.append(torch.ones(len(input_ids), dtype=torch.long))
            labels_list.append(torch.tensor(labels, dtype=torch.long))

            if "image" in sample and isinstance(sample["image"], Image.Image):
                images.append(sample["image"])
                placeholder_images.append(bool(sample.get("used_placeholder_image", False)))
            else:
                try:
                    with Image.open(sample["image_path"]) as img:
                        images.append(img.convert("RGB"))
                        placeholder_images.append(False)
                except (FileNotFoundError, UnidentifiedImageError):
                    images.append(Image.new("RGB", (224, 224), color=0))
                    placeholder_images.append(True)

        pixel_values = self.processor(images=images, return_tensors="pt")["pixel_values"]
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id or 0
        # Pad sequences to the max length in batch; labels use -100 ignore index.
        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids_list,
            batch_first=True,
            padding_value=pad_token_id,
        )
        attention_mask_padded = torch.nn.utils.rnn.pad_sequence(
            attention_masks,
            batch_first=True,
            padding_value=0,
        )
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            labels_list,
            batch_first=True,
            padding_value=-100,
        )
        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded,
            "pixel_values": pixel_values,
            "labels": labels_padded,
            "qid": qids,
            "answer_type": answer_types,
            "question": questions,
            "answer": answers,
            "content_type": content_types,
            "base_type": base_types,
            "modality": modalities,
            "triple": triples,
            "prompt_len": torch.tensor(prompt_lens, dtype=torch.long),
            "answer_len": torch.tensor(answer_lens, dtype=torch.long),
            "prompt_truncated": torch.tensor(prompt_truncated, dtype=torch.bool),
            "answer_truncated": torch.tensor(answer_truncated, dtype=torch.bool),
            "used_placeholder_image": torch.tensor(placeholder_images, dtype=torch.bool),
        }
