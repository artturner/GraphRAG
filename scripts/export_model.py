#!/usr/bin/env python3
"""Export the sentence-transformers model to .model_cache/ for Docker builds.

Run this once locally before building the Docker image:
    python scripts/export_model.py

The model is saved to .model_cache/sentence-transformers/all-MiniLM-L6-v2/
which matches the SENTENCE_TRANSFORMERS_HOME lookup path used at runtime.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_DIR = Path(".model_cache") / MODEL_NAME


def main() -> None:
    from sentence_transformers import SentenceTransformer

    print(f"Loading {MODEL_NAME} from local cache ...")
    model = SentenceTransformer(MODEL_NAME)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {OUTPUT_DIR} ...")
    model.save(str(OUTPUT_DIR))

    print(f"Done — model ready for Docker COPY at .model_cache/")


if __name__ == "__main__":
    main()
