"""Ulysses sequence parallelism for Gated DeltaNet (linear-attention) layers.

Ulysses shards each packed row along the sequence and only patches flash
attention, which all-to-alls to full-sequence/partial-head layout. A Gated
DeltaNet layer carries a recurrent state and a short causal convolution along
the sequence, so running it unpatched on a sequence shard starts every rank
after the first from a zero state: its outputs (and gradients) for that whole
shard are wrong.

The fix mirrors Ulysses attention. Heads of a Gated DeltaNet layer are
independent: the convolution is depthwise per channel, and the delta rule runs
per value head on its (grouped) key head. So all-to-all the per-token
projections to full-sequence/partial-head layout, run the unmodified
convolution and chunked delta-rule kernels on the full sequence for this
rank's heads, and all-to-all the result back before the per-token gated norm
and output projection.

Supported: transformers ``Qwen3_5GatedDeltaNet``-style modules (separate
``in_proj_qkv`` / ``in_proj_z`` / ``in_proj_b`` / ``in_proj_a``). Any other
linear-attention module under sequence parallelism fails closed.
"""

import sys
from typing import Any

import torch
import torch.nn.functional as F
from loguru import logger

from skyrl.backends.skyrl_train.distributed.ulysses.utils import (
    gather_heads_scatter_seq,
    gather_seq_scatter_heads,
    get_ulysses_sequence_parallel_rank,
    get_ulysses_sequence_parallel_world_size,
)

_REQUIRED_ATTRIBUTES = (
    "num_k_heads",
    "num_v_heads",
    "head_k_dim",
    "head_v_dim",
    "key_dim",
    "value_dim",
    "activation",
    "conv1d",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "A_log",
    "dt_bias",
    "norm",
    "out_proj",
)
_MODULE_FUNCTIONS = ("apply_mask_to_padding_states", "causal_conv1d_fn", "torch_chunk_gated_delta_rule")
# Class-name markers of sequence-mixing layers that keep state along the sequence.
_STATEFUL_MIXER_MARKERS = ("GatedDeltaNet", "LinearAttention", "Mamba", "DeltaNet", "RWKV")


class UnsupportedSequenceParallelModule(NotImplementedError):
    """A stateful sequence mixer that Ulysses sequence parallelism cannot shard."""


def _ulysses_gated_deltanet_forward(
    self,
    hidden_states: torch.Tensor,
    cache_params: Any | None = None,
    attention_mask: torch.Tensor | None = None,
    **kwargs,
):
    sp_size = get_ulysses_sequence_parallel_world_size()
    if sp_size == 1:
        return self._skyrl_original_forward(
            hidden_states, cache_params=cache_params, attention_mask=attention_mask, **kwargs
        )
    # HF models build a cache on every forward when ``config.use_cache`` is
    # true (the default), including training forwards. A cache without
    # previous state is a full-sequence pass: run the head-parallel path and
    # leave the linear-attention cache unpopulated. Incremental decoding from
    # a populated cache is not a sequence-parallel use case.
    if cache_params is not None and cache_params.has_previous_state(self.layer_idx, state_idx=0):
        raise UnsupportedSequenceParallelModule(
            "cached incremental decoding is not supported for Gated DeltaNet under Ulysses sequence parallelism"
        )
    if kwargs.get("cu_seq_lens_q") is not None:
        raise UnsupportedSequenceParallelModule(
            "cu_seq_lens_q is not supported for Gated DeltaNet under Ulysses sequence parallelism"
        )
    kernels = sys.modules[type(self).__module__]
    hidden_states = kernels.apply_mask_to_padding_states(hidden_states, attention_mask)
    batch_size, local_len, _ = hidden_states.shape
    rank = get_ulysses_sequence_parallel_rank()
    k_heads, v_heads = self.num_k_heads // sp_size, self.num_v_heads // sp_size
    k_dim, v_dim = self.head_k_dim, self.head_v_dim

    # Per-token projections on the local sequence shard, all heads.
    query, key, value = torch.split(
        self.in_proj_qkv(hidden_states), [self.key_dim, self.key_dim, self.value_dim], dim=-1
    )
    qk = torch.cat(
        [
            query.reshape(batch_size, local_len, self.num_k_heads, k_dim),
            key.reshape(batch_size, local_len, self.num_k_heads, k_dim),
        ],
        dim=-1,
    )
    vba = torch.cat(
        [
            value.reshape(batch_size, local_len, self.num_v_heads, v_dim),
            self.in_proj_b(hidden_states).unsqueeze(-1),
            self.in_proj_a(hidden_states).unsqueeze(-1),
        ],
        dim=-1,
    )

    # (bsz, seq/sp, heads, d) -> (bsz, seq, heads/sp, d). Rank r owns head block r.
    qk = gather_seq_scatter_heads(qk, seq_dim=1, head_dim=2)
    vba = gather_seq_scatter_heads(vba, seq_dim=1, head_dim=2)
    seq_len = qk.size(1)
    query, key = qk[..., :k_dim], qk[..., k_dim:]
    value, b, a = vba[..., :v_dim], vba[..., v_dim], vba[..., v_dim + 1]

    # Depthwise causal convolution over this rank's channels, in the layer's
    # original [query, key, value] channel order.
    q_channels = slice(rank * k_heads * k_dim, (rank + 1) * k_heads * k_dim)
    k_channels = slice(self.key_dim + q_channels.start, self.key_dim + q_channels.stop)
    v_channels = slice(2 * self.key_dim + rank * v_heads * v_dim, 2 * self.key_dim + (rank + 1) * v_heads * v_dim)
    weight = self.conv1d.weight.squeeze(1)
    local_weight = torch.cat([weight[q_channels], weight[k_channels], weight[v_channels]], dim=0)
    bias = self.conv1d.bias
    local_bias = None if bias is None else torch.cat([bias[q_channels], bias[k_channels], bias[v_channels]], dim=0)
    mixed = torch.cat(
        [
            query.reshape(batch_size, seq_len, -1),
            key.reshape(batch_size, seq_len, -1),
            value.reshape(batch_size, seq_len, -1),
        ],
        dim=-1,
    ).transpose(1, 2)
    mixed = kernels.causal_conv1d_fn(
        mixed.contiguous(), local_weight, local_bias, activation=self.activation, **kwargs
    ).transpose(1, 2)
    query, key, value = torch.split(mixed, [k_heads * k_dim, k_heads * k_dim, v_heads * v_dim], dim=-1)
    query = query.reshape(batch_size, seq_len, k_heads, k_dim)
    key = key.reshape(batch_size, seq_len, k_heads, k_dim)
    value = value.reshape(batch_size, seq_len, v_heads, v_dim)

    heads = slice(rank * v_heads, (rank + 1) * v_heads)
    beta = b.sigmoid()
    g = -self.A_log[heads].float().exp() * F.softplus(a.float() + self.dt_bias[heads])
    group = self.num_v_heads // self.num_k_heads
    if group > 1:
        query = query.repeat_interleave(group, dim=2)
        key = key.repeat_interleave(group, dim=2)

    kwargs.pop("cu_seq_lens_q", None)
    core_attn_out, _ = kernels.torch_chunk_gated_delta_rule(
        query,
        key,
        value,
        g=g,
        beta=beta,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=None,
        **kwargs,
    )
    # (bsz, seq, heads/sp, d) -> (bsz, seq/sp, heads, d)
    core_attn_out = gather_heads_scatter_seq(core_attn_out, seq_dim=1, head_dim=2)

    z = self.in_proj_z(hidden_states).reshape(-1, v_dim)
    core_attn_out = self.norm(core_attn_out.reshape(-1, v_dim), z)
    return self.out_proj(core_attn_out.reshape(batch_size, local_len, -1))


def _is_stateful_mixer(module: torch.nn.Module) -> bool:
    name = type(module).__name__
    return any(marker in name for marker in _STATEFUL_MIXER_MARKERS)


def patch_linear_attention_for_ulysses(model: torch.nn.Module, ulysses_sp_size: int) -> int:
    """Make every Gated DeltaNet layer of ``model`` correct under Ulysses SP.

    Returns the number of patched module classes. Raises
    ``UnsupportedSequenceParallelModule`` for any stateful sequence mixer that
    cannot be patched, rather than training on wrong log-probabilities.
    """
    if ulysses_sp_size <= 1:
        return 0
    patched = 0
    seen = set()
    for module in model.modules():
        cls = type(module)
        if cls in seen or not _is_stateful_mixer(module):
            continue
        seen.add(cls)
        kernels = sys.modules.get(cls.__module__)
        missing = [name for name in _REQUIRED_ATTRIBUTES if not hasattr(module, name)]
        missing += [name for name in _MODULE_FUNCTIONS if not hasattr(kernels, name)]
        if missing:
            raise UnsupportedSequenceParallelModule(
                f"{cls.__module__}.{cls.__name__} keeps state along the sequence and is not supported "
                f"by Ulysses sequence parallelism (missing {missing}); use sequence_parallel_size=1"
            )
        if module.num_k_heads % ulysses_sp_size or module.num_v_heads % ulysses_sp_size:
            raise UnsupportedSequenceParallelModule(
                f"{cls.__name__}: num_k_heads {module.num_k_heads} and num_v_heads {module.num_v_heads} "
                f"must both be divisible by ulysses_sp_size {ulysses_sp_size}"
            )
        if module.num_v_heads % module.num_k_heads:
            raise UnsupportedSequenceParallelModule(f"{cls.__name__}: num_v_heads must be a multiple of num_k_heads")
        if getattr(cls.forward, "_skyrl_ulysses_linear_attention", False):
            continue
        original = cls.forward
        forward = _ulysses_gated_deltanet_forward
        force_hooks = getattr(kernels, "force_accelerate_hooks", None)
        if force_hooks is not None:
            forward = force_hooks("conv1d")(forward)
        forward._skyrl_ulysses_linear_attention = True
        cls._skyrl_original_forward = original
        cls.forward = forward
        patched += 1
        logger.info(f"Patched {cls.__module__}.{cls.__name__}.forward for Ulysses sequence parallelism")
    return patched
