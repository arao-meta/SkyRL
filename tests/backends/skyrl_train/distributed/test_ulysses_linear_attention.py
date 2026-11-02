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

qwen3_5 = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
from transformers.models.qwen3_5.configuration_qwen3_5 import (
    Qwen3_5TextConfig,
)

from skyrl.backends.skyrl_train.distributed.ulysses.linear_attention import (
    UnsupportedSequenceParallelModule,
    patch_linear_attention_for_ulysses,
)
from skyrl.backends.skyrl_train.distributed.ulysses.utils import (
    set_ulysses_sequence_parallel_group,
)

SEQ_LEN = 96  # not a multiple of the 64-token delta-rule chunk


def _layer() -> torch.nn.Module:
    config = Qwen3_5TextConfig(
        hidden_size=48,
        num_hidden_layers=1,
        layer_types=["linear_attention"],
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        linear_key_head_dim=8,
        linear_value_head_dim=6,
        linear_conv_kernel_dim=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=12,
        intermediate_size=64,
        vocab_size=64,
    )
    torch.manual_seed(0)
    layer = qwen3_5.Qwen3_5GatedDeltaNet(config, layer_idx=0)
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.add_(0.1 * torch.randn_like(parameter))
    return layer


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _worker(rank: int, world_size: int, port: int, results: dict) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        set_ulysses_sequence_parallel_group(dist.group.WORLD)
        generator = torch.Generator().manual_seed(1)
        layer = _layer()
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
        results[rank] = {
            "bug_error": bug_error,
            "out_error": (out - ref_out[:, local]).abs().max().item(),
            "out_scale": ref_out.abs().max().item(),
            "input_grad_error": (local_hidden.grad - ref_hidden.grad[:, local]).abs().max().item(),
            "grad_errors": grad_errors,
        }
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4])
def test_gated_deltanet_ulysses_matches_unsharded(world_size):
    manager = mp.Manager()
    results = manager.dict()
    mp.spawn(_worker, args=(world_size, _free_port(), results), nprocs=world_size, join=True)
    for rank in range(world_size):
        result = results[rank]
        assert result["out_error"] < 1e-5 * result["out_scale"], result
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
