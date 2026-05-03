"""Pre-download the model to local disk on the SSH host.

Both colocated and PD launchers expect the model at ``$MODEL_DIR/<model_name>``.
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import qwen25_coder_7b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=None,
                    help="Override PD_BENCH_MODEL_DIR (default /data/models)")
    ap.add_argument("--mirror", default="https://hf-mirror.com",
                    help="HF endpoint (set to '' to use official)")
    args = ap.parse_args()

    cfg = qwen25_coder_7b(args.model_dir) if args.model_dir else qwen25_coder_7b()
    print(f"Target: {cfg.model_id}  →  {cfg.local_path}")

    from huggingface_hub import snapshot_download
    os.makedirs(cfg.local_path, exist_ok=True)
    kwargs = dict(repo_id=cfg.model_id, local_dir=cfg.local_path,
                  local_dir_use_symlinks=False,
                  ignore_patterns=["*.pt", "*.bin"])
    if args.mirror:
        try:
            snapshot_download(endpoint=args.mirror, **kwargs)
        except Exception as e:
            print(f"Mirror failed ({e}); trying official endpoint")
            snapshot_download(**kwargs)
    else:
        snapshot_download(**kwargs)
    print("Done.")


if __name__ == "__main__":
    main()
