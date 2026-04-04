# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
# Modifications Copyright (c) 2026 Ryota Kemmoku
# Modified from the original file in NVIDIA CUDA-QX.
# Changes made: added fast inference with KV cache and optimized repetition_penalty processing.


from transformers import GPT2LMHeadModel, GPT2Config
from torch.nn import functional as F
from torch.distributions import Categorical
import torch

from gqe_qsci.gqe.models.policy import Policy


class SmallConfig(GPT2Config):
    def __init__(self, **kwargs):
        super().__init__(n_layer=6, n_head=6, **kwargs)


class GPT2Model(GPT2LMHeadModel, Policy):
    def __init__(self, small, repetition_penalty, vocab_size, ngates):
        max_positions = int(ngates) + 1
        gpt2cfg = GPT2Config(vocab_size=vocab_size, n_positions=max_positions)
        if small:
            gpt2cfg = SmallConfig(vocab_size=vocab_size, n_positions=max_positions)
        self.repetition_penalty = repetition_penalty
        super().__init__(gpt2cfg)
        self._tril_cache = {}

    def log_prob(self, indices, temperature, return_entropy=False):
        """
        Compute next-token log-probabilities (and optionally entropies) under the same
        distribution as `act()` (i.e., includes repetition penalty + temperature).

        Args:
          indices: (B, L) token ids
          return_entropy: bool, optional, default=False

        Returns:
          - log_prob: (B, L-1)  per-step next-token log p(a_t | prefix)
          - entropy:  (B, L-1)  per-step entropy H[p(.|prefix)] (only if return_entropy=True)
        """
        out = self(indices)
        logits = out.logits  # (B, L, V)

        # GPT-2 convention: logits at position t predict token t+1, so shift.
        logits = logits[:, :-1, :]   # (B, L-1, V)
        labels = indices[:, 1:]      # (B, L-1)
        prefix = indices[:, :-1]     # (B, L-1) tokens already in the prefix at each step

        # Apply repetition penalty across all time steps at once (faster than Python loops).
        if self.repetition_penalty is not None and self.repetition_penalty > 1.0:
            logits = self._apply_repetition_penalty_sequence(
                logits=logits,
                prefix_ids=prefix,
                repetition_penalty=float(self.repetition_penalty),
            )

        log_probs = F.log_softmax(-temperature * logits, dim=-1)                  # (B, L-1, V)
        token_logp = torch.gather(log_probs, 2, labels.unsqueeze(-1)).squeeze(-1) # (B, L-1)
        if return_entropy:
            entropy = -(log_probs.exp() * log_probs).sum(dim=-1)                  # (B, L-1)
            return token_logp, entropy
        else:
            return token_logp

    def act(self, state, temperature):
        """
        Incremental decoding:
        - If state["past_key_values"] exists, use KV cache and run a forward pass on the last token only.
        - Otherwise, run a full-prefix forward pass once to initialize the cache.
        """
        idx = state["idx"]
        past = state.get("past_key_values", None)
        if past is None:
            out = self(idx, use_cache=True)
        else:
            out = self(idx[:, -1:], past_key_values=past, use_cache=True)

        # Save KV cache into state (TrainPipeline.update_state keeps non-"idx" keys intact).
        state["past_key_values"] = out.past_key_values
        logits = out.logits[:, -1, :]  # (B, V)
        if self.repetition_penalty is not None and self.repetition_penalty > 1.0:
            logits = self._apply_repetition_penalty_last(
                logits=logits,
                input_ids=idx,
                repetition_penalty=float(self.repetition_penalty),
            )
        probs = Categorical(logits=-temperature * logits)
        next_token = probs.sample()
        return next_token

    @staticmethod
    def _apply_repetition_penalty_last(logits, input_ids, repetition_penalty: float):
        """
        Last-step variant (updates logits via scatter).
        logits: (B, V)
        input_ids: (B, L_prefix)
        """
        gathered = torch.gather(logits, 1, input_ids)  # (B, L_prefix)
        updated = torch.where(
            gathered < 0,
            gathered / repetition_penalty,
            gathered * repetition_penalty,
        )
        # scatter returns a new tensor (functional-style), so this is safe.
        return logits.scatter(1, input_ids, updated)

    def _get_tril_indices(self, S: int, device: torch.device):
        """
        tril indices (ti, sj) with sj <= ti, cached.
        S: sequence length (time steps in logits)
        """
        key = (S, device.type, device.index if device.type == "cuda" else -1)
        cached = self._tril_cache.get(key)
        if cached is not None:
            return cached
        # offset=0 includes the diagonal (prefix at step t includes positions 0..t).
        ti, sj = torch.tril_indices(S, S, offset=0, device=device)
        self._tril_cache[key] = (ti, sj)
        return ti, sj

    def _apply_repetition_penalty_sequence(self, logits, prefix_ids, repetition_penalty: float):
        """
        Fast repetition-penalty application across all time steps (no Python loop).

        logits:     (B, S, V) where S = L-1 (number of next-token prediction steps)
        prefix_ids: (B, S) tokens in the prefix per step (penalize tokens in positions 0..t)
        """
        B, S, V = logits.shape
        device = logits.device

        ti, sj = self._get_tril_indices(S, device)     # (M,), (M,),  M = S(S+1)/2

        # Select which "row" (b*S + t) in the flattened logits to update.
        batch_offsets = (torch.arange(B, device=device) * S).unsqueeze(1)  # (B, 1)
        rows = (batch_offsets + ti.unsqueeze(0)).reshape(-1)              # (B*M,)

        # Select which token id to penalize (prefix position s).
        cols = prefix_ids[:, sj].reshape(-1)                               # (B*M,)

        flat = logits.reshape(B * S, V)                                    # (B*S, V)

        # Gather target elements and compute their updated values.
        vals = flat[rows, cols]                                            # (B*M,)
        updated = torch.where(
            vals < 0,
            vals / repetition_penalty,
            vals * repetition_penalty,
        )
        out = flat.clone()
        out.index_put_((rows, cols), updated, accumulate=False)

        return out.view(B, S, V)