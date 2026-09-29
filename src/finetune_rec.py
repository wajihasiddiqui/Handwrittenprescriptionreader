"""Deprecated PP-OCRv5 trainer — use LLaMA-Factory for GLM-OCR instead."""

from __future__ import annotations

import sys


def export_best() -> None:
    print("PP-OCRv5 export removed. Use: python src\\llamafactory_train.py export")
    sys.exit(1)


def main() -> None:
    print(
        "PP-OCRv5 fine-tune is removed.\n"
        "For GLM-OCR ShareGPT data:\n"
        "  python src\\prepare_finetune.py\n"
        "Train:\n"
        "  python src\\llamafactory_train.py train --mode lora"
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
