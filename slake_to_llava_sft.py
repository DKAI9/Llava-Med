"""Convert SLAKE to LLaVA-style SFT JSON."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, List

from data.slake_llava_loader import load_slake_dataset
from utils.prompting import build_image_question_text, build_slake_user_text

logger = logging.getLogger(__name__)


def to_sft_records(samples, image_root: Path, mm_use_im_start_end: bool) -> List[dict]:
    records = []
    for sample in samples:
        image_path = (image_root / "img" / sample.img_name).as_posix()
        user_text = build_slake_user_text(sample.question)
        question_text = build_image_question_text(user_text, mm_use_im_start_end)
        records.append(
            {
                "id": sample.qid,
                "image": image_path,
                "conversations": [
                    {"from": "human", "value": question_text},
                    {"from": "gpt", "value": sample.answer},
                ],
            }
        )
    return records


def write_json(path: Path, records: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(list(records), handle, ensure_ascii=False, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert SLAKE to LLaVA SFT JSON")
    parser.add_argument("--data_root", type=Path, default=Path("SLAKE"))
    parser.add_argument("--output_dir", type=Path, default=Path("work_slake_llava_med"))
    parser.add_argument("--use_hf", action="store_true")
    parser.add_argument("--mm_use_im_start_end", action="store_true", default=False)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = build_parser().parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "validation", "test"]:
        samples = load_slake_dataset(
            split=split,
            data_root=args.data_root,
            use_hf=args.use_hf,
            use_triple_context=False,
        )
        samples = sorted(samples, key=lambda s: s.qid)
        records = to_sft_records(samples, args.data_root, args.mm_use_im_start_end)
        out_path = args.output_dir / f"slake_{split}_llava_sft.json"
        write_json(out_path, records)
        logger.info("Wrote %s with %d records", out_path, len(records))


if __name__ == "__main__":
    main()
