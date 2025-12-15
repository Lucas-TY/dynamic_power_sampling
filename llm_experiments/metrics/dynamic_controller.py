from __future__ import annotations

"""
Metric-based early stopping for MH token-space sampling.

This module is separate from the paper's "Dynamic Power Sampling (DPS)" controller:
- DPS schedules the *trajectory budget* k (e.g. k ∈ {2,4,6,8}) using scalar statistics
  and intermediate hidden states.
- This file implements an optional heuristic that *stops a single MH chain early* when
  entropy/perplexity/self-confidence crosses a threshold.
"""

from dataclasses import dataclass
from typing import List

from .generation_metrics import compute_metrics_from_log_probs


@dataclass
class DynamicStopController:
    metric: str
    entropy_threshold: float = 1.0
    perplexity_threshold: float = 3.0
    self_conf_threshold: float = 0.8
    min_tokens: int = 64
    triggered: bool = False

    def should_stop(self, log_probs_norm: List[float]) -> bool:
        if len(log_probs_norm) < self.min_tokens:
            return False
        metrics = compute_metrics_from_log_probs(log_probs_norm)
        if self.metric == "entropy":
            return metrics["entropy"] <= self.entropy_threshold
        if self.metric == "perplexity":
            return metrics["perplexity"] <= self.perplexity_threshold
        if self.metric == "self_confidence":
            return metrics["self_confidence"] >= self.self_conf_threshold
        return False

    def __call__(self, log_probs_norm: List[float]) -> bool:
        stop = self.should_stop(log_probs_norm)
        if stop:
            self.triggered = True
        return stop
