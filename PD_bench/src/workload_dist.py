"""ISL/OSL distributions modelled after HumanEval and SWE-bench.

Public datasets (token counts via Qwen2 BPE tokenizer):

  HumanEval (164 problems, completion task):
      ISL  : ~140 tokens median, 50–500 range, mildly right-skewed
      OSL  : ~70  tokens median, 20–300 range
      → maps to inline_completion / function_generation

  SWE-bench Lite (300 issues, patch generation):
      ISL  : ~3500 tokens median, 800–15000 range, heavy right tail
      OSL  : ~250  tokens median, 50–1500 range
      → maps to code_explanation (long context, medium output)

Each profile draws (ISL, OSL) by sampling one of the two datasets with a
given probability, then clipping OSL to a profile-specific range. This keeps
the synthetic load anchored in the published dataset shapes rather than
arbitrary uniform ranges.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class LogNormalDist:
    """Truncated log-normal in token space."""
    median: float
    sigma:  float        # log-space std
    lo:     int
    hi:     int

    def sample(self, rng: random.Random) -> int:
        mu = math.log(self.median)
        for _ in range(8):
            v = int(round(math.exp(rng.gauss(mu, self.sigma))))
            if self.lo <= v <= self.hi:
                return v
        return max(self.lo, min(self.hi, v))


HUMANEVAL = {
    "isl": LogNormalDist(median=140, sigma=0.55, lo=50,  hi=500),
    "osl": LogNormalDist(median=70,  sigma=0.55, lo=20,  hi=300),
}
SWEBENCH = {
    "isl": LogNormalDist(median=3500, sigma=0.85, lo=800, hi=15000),
    "osl": LogNormalDist(median=250,  sigma=0.80, lo=50,  hi=1500),
}

PD_PROFILES: Dict[str, Dict] = {
    "inline_completion": {
        "datasets":   [("humaneval", 0.85), ("swebench", 0.15)],
        "osl_clip":   (10, 200),
        "TTFT_SLO":   0.5,
        "weight":     0.50,
        "description": "IDE inline completion: short-to-medium ctx, very short output",
    },
    "code_explanation": {
        "datasets":   [("swebench", 0.80), ("humaneval", 0.20)],
        "osl_clip":   (100, 600),
        "TTFT_SLO":   1.5,
        "weight":     0.25,
        "description": "Long-context explanation of a function/issue",
    },
    "function_generation": {
        "datasets":   [("humaneval", 0.60), ("swebench", 0.40)],
        "osl_clip":   (100, 800),
        "TTFT_SLO":   1.0,
        "weight":     0.25,
        "description": "Generate a complete function/class from a prompt",
    },
}

_DATASETS = {"humaneval": HUMANEVAL, "swebench": SWEBENCH}


def sample_profile(profile_name: str, rng: random.Random) -> Tuple[int, int]:
    """Draw a single (isl, osl) pair from the chosen workload profile."""
    p = PD_PROFILES[profile_name]
    names, weights = zip(*p["datasets"])
    ds = _DATASETS[rng.choices(names, weights=weights)[0]]
    isl = ds["isl"].sample(rng)
    osl = ds["osl"].sample(rng)
    lo, hi = p["osl_clip"]
    return isl, max(lo, min(hi, osl))


def sample_mixed(rng: random.Random) -> Tuple[str, int, int]:
    """Pick a profile by its production weight, then sample (isl, osl)."""
    names   = list(PD_PROFILES.keys())
    weights = [PD_PROFILES[n]["weight"] for n in names]
    profile = rng.choices(names, weights=weights)[0]
    isl, osl = sample_profile(profile, rng)
    return profile, isl, osl


def describe_distributions(n: int = 2000) -> str:
    rng = random.Random(0)
    lines = ["Workload empirical quantiles  (n=%d)" % n,
             "-" * 76]
    for name in PD_PROFILES:
        isls, osls = [], []
        for _ in range(n):
            isl, osl = sample_profile(name, rng)
            isls.append(isl); osls.append(osl)
        isls.sort(); osls.sort()
        lines.append(
            f"{name:22s}  ISL p50={isls[n//2]:5d} p95={isls[int(n*0.95)]:6d} "
            f"p99={isls[int(n*0.99)]:6d}   OSL p50={osls[n//2]:4d} "
            f"p95={osls[int(n*0.95)]:5d} p99={osls[int(n*0.99)]:5d}"
        )
    return "\n".join(lines)
