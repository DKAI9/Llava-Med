"""
Sanity-check loader + a tiny forward/generate pass for LLaVA-Med (LLaVA-Mistral 7B).

What it checks (prints to stdout):
  1) Which loader path was used (Codex wrapper -> HF / official llava fallback).
  2) Parameter counts (total vs trainable) + which submodules are trainable.
  3) Vision tower + MM projector are discoverable.
  4) The <image> token exists in the tokenizer and matches the model config (when available).
  5) A tiny generate run works end-to-end on a provided image (or a synthetic one).

Run examples:
  - Minimal (synthetic image):
      python llava_med_sanity_check.py --model_id microsoft/llava-med-v1.5-mistral-7b

  - With a real image (recommended):
      python llava_med_sanity_check.py --model_id microsoft/llava-med-v1.5-mistral-7b --image /path/to/img.png --question "What abnormality is shown?"

Notes:
  - If you use the Codex project layout, run this from the repo root (so `models/` is importable).
"""
from __future__ import annotations

import argparse
import contextlib
import os
import sys
from collections import Counter
from typing import Any, Dict, Optional, Tuple

import torch
from PIL import Image


def _choose_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str == "bf16":
        return torch.bfloat16
    if dtype_str == "fp16":
        return torch.float16
    if dtype_str == "fp32":
        return torch.float32
    raise ValueError(f"Unknown --dtype {dtype_str!r}. Use bf16|fp16|fp32.")


def _count_params(model: torch.nn.Module) -> Tuple[int, int]:
    total = 0
    trainable = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return total, trainable


def _fmt_int(n: int) -> str:
    return f"{n:,}"


def _device_for_inputs(model: torch.nn.Module, prefer: str) -> torch.device:
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "cpu":
        return torch.device("cpu")
    for p in model.parameters():
        return p.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _find_attr_chain(obj: Any, chains: Tuple[str, ...]) -> Optional[Any]:
    """Try attribute names; supports dotted chains like 'model.mm_projector'."""
    for chain in chains:
        cur = obj
        ok = True
        for part in chain.split("."):
            if not hasattr(cur, part):
                ok = False
                break
            cur = getattr(cur, part)
        if ok and cur is not None:
            return cur
    return None


def _summarize_trainables(model: torch.nn.Module, topk: int = 12) -> None:
    bucket = Counter()
    for name, p in model.named_parameters():
        if p.requires_grad:
            prefix = name.split(".")[0]
            bucket[prefix] += p.numel()
    if not bucket:
        print("Trainable modules: (none)")
        return
    print("Trainable modules (by top-level prefix):")
    for k, v in bucket.most_common(topk):
        print(f"  - {k:24s} {_fmt_int(v)} params")
    if len(bucket) > topk:
        print(f"  ... ({len(bucket)-topk} more)")


def _make_synthetic_image(size: int = 336) -> Image.Image:
    import numpy as np

    x = np.linspace(0, 255, size, dtype=np.uint8)
    img = np.stack(
        [
            x[None, :].repeat(size, 0),
            x[:, None].repeat(size, 1),
            (255 - x)[None, :].repeat(size, 0),
        ],
        axis=-1,
    )
    return Image.fromarray(img, mode="RGB")


def _load_via_codex_wrapper(model_id: str, device_map: str, use_flash_attention: bool, dtype: torch.dtype):
    try:
        from models.llava_med_trainable import LlavaMedTrainable  # type: ignore
    except Exception as exc:
        return None, None, f"codex_wrapper_import_failed: {exc}"

    try:
        m = LlavaMedTrainable(
            model_id=model_id,
            device_map=device_map,
            use_flash_attention=use_flash_attention,
            dtype=dtype,
        )
        return m.processor, m.model, "codex_wrapper"
    except TypeError:
        m = LlavaMedTrainable(
            model_id=model_id,
            device_map=device_map,
            use_flash_attention=use_flash_attention,
        )
        return m.processor, m.model, "codex_wrapper(no_dtype_arg)"


def _load_via_hf(model_id: str, device_map: str, use_flash_attention: bool, dtype: torch.dtype):
    import transformers
    from transformers import AutoModelForVision2Seq, AutoProcessor

    LlavaForConditionalGeneration = getattr(transformers, "LlavaForConditionalGeneration", None)

    kwargs: Dict[str, Any] = {"device_map": device_map, "torch_dtype": dtype}
    if use_flash_attention:
        kwargs["attn_implementation"] = "flash_attention_2"

    processor = AutoProcessor.from_pretrained(model_id)

    loading_info = None
    if LlavaForConditionalGeneration is not None:
        out = LlavaForConditionalGeneration.from_pretrained(model_id, output_loading_info=True, **kwargs)
    else:
        out = AutoModelForVision2Seq.from_pretrained(model_id, output_loading_info=True, **kwargs)

    if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict):
        model, loading_info = out
    else:
        model = out  # type: ignore

    return processor, model, loading_info, "hf_transformers"


def _diagnose_loading_info(loading_info: Optional[dict]) -> None:
    if not loading_info:
        print("Loading info: (not available)")
        return
    unexpected = loading_info.get("unexpected_keys") or []
    missing = loading_info.get("missing_keys") or []
    mismatched = loading_info.get("mismatched_keys") or []
    print(f"Loading info: unexpected={len(unexpected)}, missing={len(missing)}, mismatched={len(mismatched)}")
    if unexpected:
        prefixes = Counter(k.split(".")[0] for k in unexpected)
        top = prefixes.most_common(8)
        print("  Unexpected key prefixes (top):", ", ".join(f"{k}:{v}" for k, v in top))
    if missing:
        prefixes = Counter(k.split(".")[0] for k in missing)
        top = prefixes.most_common(8)
        print("  Missing key prefixes (top):   ", ", ".join(f"{k}:{v}" for k, v in top))
    if mismatched:
        print("  Mismatched keys (first 5):")
        for item in mismatched[:5]:
            print("   -", item)


def _check_image_token(processor: Any, model: torch.nn.Module) -> None:
    tok = getattr(processor, "tokenizer", None)
    if tok is None:
        print("Tokenizer: (not found on processor)")
        return

    image_token = "<image>"
    tok_id = tok.convert_tokens_to_ids(image_token)
    print(f"Tokenizer: {type(tok).__name__} | '<image>' token id: {tok_id}")

    cfg = getattr(model, "config", None)
    cfg_id = getattr(cfg, "image_token_index", None) if cfg is not None else None
    if cfg_id is not None:
        print(f"Model config image_token_index: {cfg_id} | matches tokenizer: {bool(tok_id == cfg_id)}")


def _check_vision_and_projector(model: torch.nn.Module) -> Tuple[Optional[torch.nn.Module], Optional[torch.nn.Module]]:
    vision = _find_attr_chain(model, ("vision_tower", "vision_model", "model.vision_tower", "model.vision_model"))
    if vision is None:
        get_vision = getattr(model, "get_vision_tower", None)
        if callable(get_vision):
            with contextlib.suppress(Exception):
                vision = get_vision()

    proj = _find_attr_chain(
        model,
        ("multi_modal_projector", "mm_projector", "model.mm_projector", "model.multi_modal_projector"),
    )

    def _module_summary(tag: str, m: Optional[torch.nn.Module]) -> None:
        if m is None:
            print(f"{tag}: NOT FOUND")
            return
        total, trainable = _count_params(m)
        print(f"{tag}: {type(m).__name__} | params={_fmt_int(total)} | trainable={_fmt_int(trainable)}")

    _module_summary("Vision tower", vision)
    _module_summary("MM projector", proj)
    return vision, proj


def _to_device(obj: Any, device: torch.device) -> Any:
    if hasattr(obj, "to"):
        return obj.to(device)
    return obj


def _normalize_official_images(images: Any, device: torch.device) -> Any:
    # ProcessorShim via process_images may return list[tensor] or tensor.
    if isinstance(images, list):
        if len(images) == 1:
            t = images[0]
            if isinstance(t, torch.Tensor) and t.ndim == 3:
                t = t.unsqueeze(0)
            return t.to(device=device, dtype=torch.float16)
        # try stacking if possible
        if all(isinstance(t, torch.Tensor) for t in images):
            ts = []
            for t in images:
                if t.ndim == 3:
                    t = t.unsqueeze(0)
                ts.append(t)
            try:
                return torch.cat(ts, dim=0).to(device=device, dtype=torch.float16)
            except Exception:
                return [t.to(device=device, dtype=torch.float16) for t in images]
        return images
    if isinstance(images, torch.Tensor):
        return images.to(device=device, dtype=torch.float16)
    return images


def _tiny_generate(processor: Any, model: torch.nn.Module, image: Image.Image, question: str, device_pref: str, max_new_tokens: int) -> None:
    model.eval()
    device = _device_for_inputs(model, device_pref)
    print(f"Using device for inputs: {device}")

    is_official = bool(getattr(processor, "is_llava_official", False))

    if not is_official:
        inputs = processor(text=f"<image>\n{question}", images=image, return_tensors="pt")
        inputs = {k: _to_device(v, device) for k, v in dict(inputs).items()}
        if "pixel_values" in inputs and device.type == "cuda":
            pv = inputs["pixel_values"]
            if pv.dtype not in (torch.float16, torch.bfloat16):
                inputs["pixel_values"] = pv.to(dtype=torch.float16)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        tok = getattr(processor, "tokenizer", None)
        txt = tok.batch_decode(out, skip_special_tokens=True)[0] if tok is not None else str(out)
        print("=== GENERATION (HF) ===")
        print(txt.strip())
        return

    # Official llava prompt builder (matches common llava_v1-style inference) :contentReference[oaicite:0]{index=0}
    try:
        from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from llava.conversation import conv_templates
        from llava.mm_utils import tokenizer_image_token
    except Exception as exc:
        print("Official loader detected but required llava imports failed:", exc)
        return

    conv_mode = "llava_v1"
    if conv_mode not in conv_templates:
        conv_mode = next(iter(conv_templates.keys()))
    conv = conv_templates[conv_mode].copy()

    inp = DEFAULT_IMAGE_TOKEN + "\n" + question
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    tok = getattr(processor, "tokenizer", None)
    if tok is None:
        print("ProcessorShim tokenizer missing; cannot build input ids.")
        return

    input_ids = tokenizer_image_token(prompt, tok, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

    img_pack = processor([image], return_tensors="pt")["pixel_values"]
    images = _normalize_official_images(img_pack, device)

    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
    with torch.inference_mode():
        try:
            out = model.generate(input_ids, images=images, image_sizes=[image.size], **gen_kwargs)
        except TypeError:
            out = model.generate(input_ids, images=images, **gen_kwargs)

    txt = tok.batch_decode(out, skip_special_tokens=True)[0]
    print("=== GENERATION (OFFICIAL LLaVA) ===")
    print(txt.strip())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, default="microsoft/llava-med-v1.5-mistral-7b")
    ap.add_argument("--device_map", type=str, default="auto")
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--use_flash_attention", action="store_true")
    ap.add_argument("--skip_generate", action="store_true")
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--image", type=str, default="")
    ap.add_argument("--question", type=str, default="Describe the key finding in this medical image.")
    ap.add_argument("--no_codex_wrapper", action="store_true")
    args = ap.parse_args()

    dtype = _choose_dtype(args.dtype)
    if torch.cuda.is_available():
        os.environ.setdefault("TORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    print("=== LLaVA-Med Sanity Check ===")
    print("model_id:", args.model_id)
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("cuda device:", torch.cuda.get_device_name(0))

    processor = None
    model = None
    loader = None
    loading_info = None

    if not args.no_codex_wrapper:
        processor, model, loader = _load_via_codex_wrapper(
            model_id=args.model_id,
            device_map=args.device_map,
            use_flash_attention=args.use_flash_attention,
            dtype=dtype,
        )
        if processor is None or model is None:
            print("Codex wrapper load failed:", loader)

    if processor is None or model is None:
        print("Falling back to direct HF transformers load...")
        processor, model, loading_info, loader = _load_via_hf(
            model_id=args.model_id,
            device_map=args.device_map,
            use_flash_attention=args.use_flash_attention,
            dtype=dtype,
        )

    print("Loader used:", loader)
    _diagnose_loading_info(loading_info)

    total, trainable = _count_params(model)
    print(f"Params: total={_fmt_int(total)} | trainable={_fmt_int(trainable)} | trainable%={(100.0*trainable/total if total else 0):.6f}%")
    _summarize_trainables(model)

    _check_image_token(processor, model)
    _check_vision_and_projector(model)

    if args.skip_generate:
        print("Skipping generate (as requested).")
        return 0

    img = Image.open(args.image).convert("RGB") if args.image else _make_synthetic_image(336)
    print(f"Image size used: {img.size}")
    print(f"Question: {args.question}")

    try:
        _tiny_generate(processor, model, img, args.question, args.device, args.max_new_tokens)
    except RuntimeError as exc:
        print("Generate failed with RuntimeError:")
        print(exc)
        if torch.cuda.is_available():
            try:
                free, total_mem = torch.cuda.mem_get_info()
                print(f"CUDA mem free={free/1024**3:.2f} GiB / total={total_mem/1024**3:.2f} GiB")
            except Exception:
                pass
        return 2

    print("Sanity check finished OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
