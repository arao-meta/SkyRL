"""Gated DeltaNet under Ulysses sequence parallelism matches the unsharded layer.

Runs real multi-process collectives on CPU (gloo) with the transformers
``Qwen3_5GatedDeltaNet`` layer and its torch reference kernels.
"""

import copy
import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import DynamicCache

qwen3_5 = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
from transformers.models.qwen3_5.configuration_qwen3_5 import (
    Qwen3_5TextConfig,
)

from skyrl.backends.skyrl_train.distributed.ulysses.linear_attention import (
    UnsupportedSequenceParallelModule,
    _is_stateful_mixer,
    check_packed_rows_supported,
    has_stateful_mixers,
    patch_linear_attention_for_ulysses,
)
from skyrl.backends.skyrl_train.distributed.ulysses.utils import (
    set_ulysses_sequence_parallel_group,
)

SEQ_LEN = 96  # not a multiple of the 64-token delta-rule chunk


def _config(v_heads: int = 8) -> Qwen3_5TextConfig:
    return Qwen3_5TextConfig(
        hidden_size=48,
        num_hidden_layers=1,
        layer_types=["linear_attention"],
        linear_num_key_heads=4,
        linear_num_value_heads=v_heads,
        linear_key_head_dim=8,
        linear_value_head_dim=6,
        linear_conv_kernel_dim=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=12,
        intermediate_size=64,
        vocab_size=64,
    )


def _layer(v_heads: int = 8) -> torch.nn.Module:
    torch.manual_seed(0)
    layer = qwen3_5.Qwen3_5GatedDeltaNet(_config(v_heads), layer_idx=0)
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.add_(0.1 * torch.randn_like(parameter))
    return layer


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _worker(rank: int, world_size: int, port: int, results: dict, v_heads: int = 8) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        set_ulysses_sequence_parallel_group(dist.group.WORLD)
        generator = torch.Generator().manual_seed(1)
        layer = _layer(v_heads)
        reference = copy.deepcopy(layer)
        hidden = torch.randn(1, SEQ_LEN, 48, generator=generator)
        upstream = torch.randn(1, SEQ_LEN, 48, generator=generator)
        local = slice(rank * SEQ_LEN // world_size, (rank + 1) * SEQ_LEN // world_size)

        # Reference: one process, full sequence.
        ref_hidden = hidden.clone().requires_grad_()
        ref_out = reference(ref_hidden)
        (ref_out * upstream).sum().backward()

        # Unpatched layer on a sequence shard (the bug).
        with torch.no_grad():
            buggy = layer(hidden[:, local])
        bug_error = (buggy - ref_out[:, local]).abs().max().item()

        assert patch_linear_attention_for_ulysses(layer, world_size) == 1
        local_hidden = hidden[:, local].clone().requires_grad_()
        out = layer(local_hidden)
        (out * upstream[:, local]).sum().backward()
        grad_errors = {}
        for (name, parameter), (_, ref_parameter) in zip(layer.named_parameters(), reference.named_parameters()):
            grad = parameter.grad.clone()
            dist.all_reduce(grad)  # FSDP/DDP sum gradient contributions across SP ranks
            grad_errors[name] = ((grad - ref_parameter.grad).abs().max() / ref_parameter.grad.abs().max()).item()
        # HF models pass a fresh cache on every forward when use_cache is true
        # (the default), including training forwards; it must not bypass the fix.
        with torch.no_grad():
            cached = layer(hidden[:, local], cache_params=DynamicCache(config=_config(v_heads)))
        results[rank] = {
            "cached_error": (cached - ref_out[:, local]).abs().max().item(),
            "bug_error": bug_error,
            "out_error": (out - ref_out[:, local]).abs().max().item(),
            "out_scale": ref_out.abs().max().item(),
            "input_grad_error": (local_hidden.grad - ref_hidden.grad[:, local]).abs().max().item(),
            "grad_errors": grad_errors,
        }
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size,v_heads", [(2, 8), (4, 8), (4, 12)])  # GQA ratio 2 and 3
def test_gated_deltanet_ulysses_matches_unsharded(world_size, v_heads):
    with mp.Manager() as manager:
        shared = manager.dict()
        mp.spawn(_worker, args=(world_size, _free_port(), shared, v_heads), nprocs=world_size, join=True)
        results = dict(shared)
    for rank in range(world_size):
        result = results[rank]
        assert result["out_error"] < 1e-5 * result["out_scale"], result
        assert result["cached_error"] < 1e-5 * result["out_scale"], result
        assert result["input_grad_error"] < 1e-5, result
        assert max(result["grad_errors"].values()) < 1e-4, result
    # Without the patch every shard after the first starts from zero state.
    assert results[0]["bug_error"] < 1e-5
    assert all(results[rank]["bug_error"] > 1e-3 for rank in range(1, world_size))


def test_sp1_is_a_noop():
    layer = _layer()
    forward = type(layer).forward
    assert patch_linear_attention_for_ulysses(layer, 1) == 0
    assert type(layer).forward is forward


def test_unsupported_stateful_mixer_fails_closed():
    class FakeGatedDeltaNet(torch.nn.Module):
        def forward(self, hidden_states):
            return hidden_states

    model = torch.nn.Sequential(FakeGatedDeltaNet())
    with pytest.raises(UnsupportedSequenceParallelModule, match="not supported"):
        patch_linear_attention_for_ulysses(model, 2)


def test_indivisible_heads_fail_closed():
    layer = _layer()  # 4 key heads, 8 value heads
    with pytest.raises(UnsupportedSequenceParallelModule, match="divisible"):
        patch_linear_attention_for_ulysses(layer, 3)


class _FakeCache:
    def __init__(self, populated):
        self.populated = populated

    def has_previous_state(self, layer_idx, state_idx=0):
        return self.populated


def _patched_layer_under_fake_sp(monkeypatch):
    from skyrl.backends.skyrl_train.distributed.ulysses import linear_attention

    layer = _layer()
    original = type(layer).forward
    monkeypatch.setattr(type(layer), "forward", original)  # restored after the test
    monkeypatch.setattr(linear_attention, "get_ulysses_sequence_parallel_world_size", lambda: 2)
    assert patch_linear_attention_for_ulysses(layer, 2) == 1
    return layer


@pytest.mark.parametrize("kwargs", [{"cu_seq_lens_q": torch.tensor([0, 4])}, {"seq_idx": torch.zeros(1, 4)}])
def test_boundary_kwargs_fail_closed_under_sp(monkeypatch, kwargs):
    layer = _patched_layer_under_fake_sp(monkeypatch)
    with pytest.raises(UnsupportedSequenceParallelModule, match=next(iter(kwargs))):
        layer(torch.randn(1, 4, 48), **kwargs)


def test_cached_decoding_fails_closed_under_sp(monkeypatch):
    layer = _patched_layer_under_fake_sp(monkeypatch)
    with pytest.raises(UnsupportedSequenceParallelModule, match="incremental decoding"):
        layer(torch.randn(1, 1, 48), cache_params=_FakeCache(populated=True))


def test_apply_monkey_patch_patches_gated_deltanet_in_a_model(monkeypatch):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    from skyrl.backends.skyrl_train.distributed.ulysses import monkey_patch

    config = _config()
    config.num_hidden_layers = 2
    config.layer_types = ["linear_attention", "full_attention"]
    model = Qwen3_5ForCausalLM(config)
    cls = type(model.model.layers[0].linear_attn)
    monkeypatch.setattr(cls, "forward", cls.forward)  # restored after the test
    from transformers.integrations import flash_attention

    monkeypatch.setattr(flash_attention, "_flash_attention_forward", flash_attention._flash_attention_forward)
    monkey_patch.apply_monkey_patch(model=model, ulysses_sp_size=2)
    assert getattr(cls.forward, "_skyrl_ulysses_linear_attention", False)


def test_stateful_mixer_detection_is_structural_and_precise():
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    config = _config()
    config.num_hidden_layers = 2
    config.layer_types = ["linear_attention", "full_attention"]
    model = Qwen3_5ForCausalLM(config)
    flagged = {type(m).__name__ for m in model.modules() if _is_stateful_mixer(m)}
    assert flagged == {"Qwen3_5GatedDeltaNet"}
    assert has_stateful_mixers(model)

    class ShortConvMixer(torch.nn.Module):  # no telltale class name
        def forward(self, hidden_states, cache_params=None):
            return hidden_states

    assert _is_stateful_mixer(ShortConvMixer())
    assert not has_stateful_mixers(torch.nn.Sequential(torch.nn.Linear(2, 2)))


def test_packed_rows_with_several_sequences_fail_closed():
    check_packed_rows_supported(True, 1)
    check_packed_rows_supported(False, 4)
    with pytest.raises(UnsupportedSequenceParallelModule, match="micro batch size of 1"):
        check_packed_rows_supported(True, 2)
