"""Tokenizer-precise prompt generator. Same tokenizer-driven approach as the
single-GPU benchmark, but driven by the HumanEval/SWE-bench distributions
rather than fixed uniform ranges."""
from __future__ import annotations

import random
from typing import List

from transformers import AutoTokenizer


_SEED_CODE = """# Data processing pipeline
import numpy as np
import pandas as pd
from typing import List, Dict, Optional

def process_records(records: List[Dict], threshold: float = 0.5) -> pd.DataFrame:
    \"\"\"Filter and aggregate records above threshold.\"\"\"
    df = pd.DataFrame(records)
    filtered = df[df['score'] > threshold]
    return filtered.groupby('category').agg({
        'value': ['mean', 'std', 'count'],
        'timestamp': 'max'
    }).reset_index()

class AnomalyDetector:
    def __init__(self, window_size: int = 100, sigma: float = 3.0):
        self.window_size = window_size
        self.sigma = sigma
        self.history = []

    def detect(self, value: float) -> bool:
        if len(self.history) < self.window_size:
            self.history.append(value)
            return False
        recent = self.history[-self.window_size:]
        mean = np.mean(recent)
        std = np.std(recent)
        is_anomaly = abs(value - mean) > self.sigma * std
        self.history.append(value)
        return is_anomaly

def parse_log_entry(line: str) -> Optional[Dict]:
    parts = line.strip().split('|')
    if len(parts) < 3:
        return None
    return {
        'timestamp': parts[0],
        'level': parts[1],
        'message': '|'.join(parts[2:])
    }
"""


class WorkloadGenerator:
    def __init__(self, tokenizer_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True
        )
        self._seed_tokens = self.tokenizer.encode(_SEED_CODE, add_special_tokens=False)

    def generate_prompt(self, isl: int, seed: int = None) -> str:
        """Build a prompt of exactly ``isl`` tokens (±2). Unique prefix on every
        call defeats vLLM prefix-cache reuse during TTFT measurement."""
        if seed is None:
            seed = random.randint(0, 10**9)
        rng = random.Random(seed)
        unique_prefix = (
            f"# Request {rng.randint(10**8, 10**9)} "
            f"context {rng.choice(['alpha', 'beta', 'gamma', 'delta'])}\n"
        )
        prefix_tokens = self.tokenizer.encode(unique_prefix, add_special_tokens=False)
        base = prefix_tokens + self._seed_tokens
        if len(base) >= isl:
            target = base[:isl]
        else:
            repeats = (isl // len(base)) + 1
            target = (base * repeats)[:isl]
        return self.tokenizer.decode(target, skip_special_tokens=True)
