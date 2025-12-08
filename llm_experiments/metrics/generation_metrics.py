import math
from typing import Iterable, List, Dict

import torch
import torch.nn.functional as F


def compute_generation_metrics(scores: Iterable[torch.Tensor], generated_ids: torch.Tensor) -> Dict[str, float]:
    """
    Compute entropy, perplexity, and self-confidence statistics from generation scores.
    Args:
        scores: iterable of per-step logits (as returned by HF generate with output_scores=True)
        generated_ids: tensor of chosen token ids (after removing prompt)
    """
    score_list: List[torch.Tensor] = [s.detach().to("cpu") for s in scores]
    if not score_list:
        return {
            "entropy": math.nan,
            "perplexity": math.nan,
            "self_confidence": math.nan,
            "entropy_series": [],
            "perplexity_series": [],
            "self_confidence_series": [],
        }

    score_tensor = torch.stack(score_list, dim=0).to(torch.float32)
    # HF generate returns scores with shape [toks, batch, vocab]; we only decode one sample,
    # so drop the redundant batch dimension when it is present to keep shapes aligned.
    if score_tensor.dim() == 3:
        if score_tensor.size(1) != 1:
            raise ValueError(
                "compute_generation_metrics expected batch size 1, got "
                f"{score_tensor.size(1)}."
            )
        score_tensor = score_tensor[:, 0, :]
    token_ids = generated_ids.detach().to(torch.long).reshape(-1)
    steps = min(score_tensor.size(0), token_ids.size(0))
    if steps == 0:
        return {
            "entropy": math.nan,
            "perplexity": math.nan,
            "self_confidence": math.nan,
            "entropy_series": [],
            "perplexity_series": [],
            "self_confidence_series": [],
        }
    score_tensor = score_tensor[:steps]
    token_ids = token_ids[:steps]
    probs = F.softmax(score_tensor, dim=-1)
    entropy_per_token = (-probs * probs.clamp_min(1e-12).log()).sum(dim=-1)

    log_probs = F.log_softmax(score_tensor, dim=-1)
    idx = token_ids.unsqueeze(-1)
    chosen_log_probs = torch.gather(log_probs, dim=-1, index=idx).squeeze(-1)
    neg_log_prob = -chosen_log_probs
    perplexity_per_token = torch.exp(neg_log_prob)

    max_conf = probs.max(dim=-1).values

    return {
        "entropy": entropy_per_token.mean().item(),
        "perplexity": perplexity_per_token.mean().item(),
        "self_confidence": max_conf.mean().item(),
        "entropy_series": entropy_per_token.tolist(),
        "perplexity_series": perplexity_per_token.tolist(),
        "self_confidence_series": max_conf.tolist(),
    }


def compute_metrics_from_log_probs(log_probs: Iterable[float]) -> Dict[str, float]:
    """Approximate metrics when only per-token log probabilities are available."""
    log_prob_tensor = torch.tensor(list(log_probs), dtype=torch.float32)
    if log_prob_tensor.numel() == 0:
        return {
            "entropy": math.nan,
            "perplexity": math.nan,
            "self_confidence": math.nan,
            "entropy_series": [],
            "perplexity_series": [],
            "self_confidence_series": [],
        }
    neg_log_probs = -log_prob_tensor
    entropy_series = neg_log_probs.tolist()
    perplexity_series = torch.exp(neg_log_probs).tolist()
    self_conf_series = torch.exp(log_prob_tensor).tolist()
    avg_neg_log_prob = neg_log_probs.mean().item()
    return {
        "entropy": avg_neg_log_prob,
        "perplexity": float(math.exp(avg_neg_log_prob)),
        "self_confidence": torch.exp(log_prob_tensor).mean().item(),
        "entropy_series": entropy_series,
        "perplexity_series": perplexity_series,
        "self_confidence_series": self_conf_series,
    }
