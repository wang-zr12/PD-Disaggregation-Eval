"""Workload profiles, prompt generation, and request sampling."""
from __future__ import annotations

import random
from typing import List

from transformers import AutoTokenizer


WORKLOAD_PROFILES = {
    "inline_completion": {
        "ISL_range": (512, 2048),
        "OSL_range": (20, 150),
        "weight":    0.50,
        "TTFT_SLO":  0.5,
        "description": "Paste file context, complete a small amount of code",
    },
    "code_explanation": {
        "ISL_range": (256, 1024),
        "OSL_range": (150, 400),
        "weight":    0.25,
        "TTFT_SLO":  1.5,
        "description": "Explain the function/logic of code",
    },
    "function_generation": {
        "ISL_range": (128, 512),
        "OSL_range": (200, 600),
        "weight":    0.25,
        "TTFT_SLO":  0.5,
        "description": "Generate a complete function or class from a comment",
    },
}


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
    """Builds prompts of an exact token length and samples request batches."""

    def __init__(self, tokenizer_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True
        )
        self._seed_tokens = self.tokenizer.encode(_SEED_CODE, add_special_tokens=False)

    def generate_prompt(self, isl: int, seed: int = None) -> str:
        """Build a prompt of exactly ``isl`` tokens (±2). A unique prefix is added
        on every call to defeat vLLM prefix-cache reuse during TTFT measurement."""
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

    def sample_workload(self, profile_name: str, n: int, seed: int = 42) -> List[dict]:
        rng = random.Random(seed)
        profile = WORKLOAD_PROFILES[profile_name]
        out = []
        for _ in range(n):
            isl = rng.randint(*profile["ISL_range"])
            osl = rng.randint(*profile["OSL_range"])
            out.append({
                "prompt":     self.generate_prompt(isl, seed=rng.randint(0, 10**9)),
                "max_tokens": osl,
                "target_isl": isl,
                "target_osl": osl,
                "profile":    profile_name,
            })
        return out

    def sample_mixed_workload(self, n: int, seed: int = 42) -> List[dict]:
        rng = random.Random(seed)
        names   = list(WORKLOAD_PROFILES.keys())
        weights = [WORKLOAD_PROFILES[p]["weight"] for p in names]
        out = []
        for _ in range(n):
            name = rng.choices(names, weights=weights)[0]
            p    = WORKLOAD_PROFILES[name]
            isl  = rng.randint(*p["ISL_range"])
            osl  = rng.randint(*p["OSL_range"])
            out.append({
                "prompt":     self.generate_prompt(isl, seed=rng.randint(0, 10**9)),
                "max_tokens": osl,
                "target_isl": isl,
                "target_osl": osl,
                "profile":    name,
            })
        return out

    def sample_fixed(self, isl: int, osl: int, n: int, profile: str = "fixed",
                    seed: int = 0) -> List[dict]:
        """Fixed-shape requests for ISL/OSL sweeps. Each request gets its own
        prompt seed so the prefix cache cannot help."""
        return [{
            "prompt":     self.generate_prompt(isl, seed=seed * 10000 + i),
            "max_tokens": osl,
            "target_isl": isl,
            "target_osl": osl,
            "profile":    profile,
        } for i in range(n)]
