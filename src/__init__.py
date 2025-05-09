"""
DeepSeek Multi-Latent Attention Implementation
Copyright (c) 2025

Implementation of the Multi-Latent Attention mechanism from the DeepSeek-V2 paper.
"""

from .mla import MultiHeadLatentAttention, precompute_freqs_cis, reshape_for_broadcast, apply_rotary_emb

__version__ = "0.1.0"
__all__ = ["MultiHeadLatentAttention", "precompute_freqs_cis", "reshape_for_broadcast","apply_rotary_emb"]