"""Export the already-trained PP-OCRv5 reader to models/medical_rec_infer."""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from finetune_rec import export_best

if __name__ == "__main__":
    export_best()
