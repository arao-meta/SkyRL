from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pytest

from skyrl.train.fully_async_trainer import FullyAsyncRayPPOTrainer
from skyrl.train.trainer import RayPPOTrainer, resolve_scheduler_total_training_steps
from skyrl.train.utils.trainer_utils import cleanup_old_checkpoints


@dataclass
class FakeModelConfig:
    path: str | None = None


@dataclass
class FakeCriticConfig:
    model: FakeModelConfig


@dataclass
class FakeTrainerConfig:
    ckpt_path: str
    max_ckpts_to_keep: int
    critic: FakeCriticConfig


@dataclass
class FakeConfig:
    trainer: FakeTrainerConfig


class FakeLoader:
    def state_dict(self):
        return {"cursor": 7}


class FakeAsyncLoader(FakeLoader):
    def get_consumed_uids_list(self):
        return ["group-1"]

    def get_filtered_uids_list(self):
        return ["group-dropped"]


class FakeDispatch:
    def __init__(self, checkpoint_root: Path, *, require_async_state: bool = False):
        self.checkpoint_root = checkpoint_root
        self.require_async_state = require_async_state
        self.events = []

    def save_checkpoint(self, model, target, tokenizer):
        del tokenizer
        self.events.append(f"save:{model}")
        path = Path(target)
        path.mkdir(parents=True, exist_ok=True)
        (path / "rank-0.pt").write_bytes(b"model-state")

    def finalize_pending_saves(self, model):
        if self.require_async_state:
            assert (self.checkpoint_root / "global_step_3" / "fully_async_state.pt").is_file()
        self.events.append(f"finalize:{model}")


def make_trainer(tmp_path: Path, *, fully_async: bool = False):
    cls = FullyAsyncRayPPOTrainer if fully_async else RayPPOTrainer
    trainer = cls.__new__(cls)
    trainer.cfg = FakeConfig(
        trainer=FakeTrainerConfig(
            ckpt_path=str(tmp_path),
            max_ckpts_to_keep=3,
            critic=FakeCriticConfig(model=FakeModelConfig()),
        )
    )
    trainer.global_step = 3
    trainer.epoch = 1
    trainer.tokenizer = None
    trainer.train_dataloader = FakeAsyncLoader() if fully_async else FakeLoader()
    if fully_async:
        trainer.async_train_dataloader = trainer.train_dataloader
    trainer.dispatch = FakeDispatch(tmp_path, require_async_state=fully_async)
    trainer.all_timings = defaultdict(float)
    trainer.cleaned = 0

    def cleanup():
        trainer.cleaned += 1

    trainer._cleanup_old_checkpoints = cleanup
    return trainer


@pytest.mark.parametrize(
    "phase,receipt_expected,latest_expected",
    [
        ("after_additional_state", False, False),
        ("after_backend_finalize", False, False),
        ("after_readback_validation", False, False),
        ("after_checkpoint_receipt", True, False),
        ("after_latest_marker", True, True),
    ],
)
def test_checkpoint_failure_boundaries_are_retryable(
    tmp_path, monkeypatch, phase, receipt_expected, latest_expected
):
    trainer = make_trainer(tmp_path)
    monkeypatch.setenv("SKYRL_CHECKPOINT_FAIL_PHASE", phase)
    with pytest.raises(RuntimeError, match=phase):
        trainer.save_checkpoints()

    checkpoint = tmp_path / "global_step_3"
    assert (checkpoint / "checkpoint_complete.json").is_file() is receipt_expected
    assert (tmp_path / "latest_ckpt_global_step.txt").is_file() is latest_expected
    assert trainer.cleaned == 0

    monkeypatch.delenv("SKYRL_CHECKPOINT_FAIL_PHASE")
    assert trainer.save_checkpoints() == str(checkpoint)
    receipt = json.loads((checkpoint / "checkpoint_complete.json").read_text())
    assert receipt["global_step"] == 3
    assert (tmp_path / "latest_ckpt_global_step.txt").read_text() == "3"
    assert trainer.cleaned == 1


def test_fully_async_state_precedes_finalize_and_receipt(tmp_path):
    trainer = make_trainer(tmp_path, fully_async=True)
    checkpoint = Path(trainer.save_checkpoints())
    assert trainer.dispatch.events == ["save:policy", "finalize:policy"]
    state = __import__("torch").load(
        checkpoint / "fully_async_state.pt", map_location="cpu", weights_only=False
    )
    assert state == {
        "consumed_uids": ["group-1"],
        "filtered_uids": ["group-dropped"],
        "epoch": 1,
        "global_step": 3,
    }
    receipt = json.loads((checkpoint / "checkpoint_complete.json").read_text())
    names = {member["path"] for member in receipt["members"]}
    assert {"data.pt", "trainer_state.pt", "fully_async_state.pt", "policy/rank-0.pt"} <= names

    (checkpoint / "unexpected").write_text("late mutation")
    with pytest.raises(RuntimeError, match="member set"):
        trainer._validate_checkpoint_receipt(str(checkpoint))


def test_checkpoint_cleanup_ignores_incomplete_directories(tmp_path):
    for step in (1, 2, 3):
        checkpoint = tmp_path / f"global_step_{step}"
        checkpoint.mkdir()
        if step != 1:
            (checkpoint / "checkpoint_complete.json").write_text("{}")

    cleanup_old_checkpoints(
        str(tmp_path), 1, require_complete_receipt=True
    )

    assert (tmp_path / "global_step_1").is_dir()
    assert not (tmp_path / "global_step_2").exists()
    assert (tmp_path / "global_step_3").is_dir()


def test_scheduler_horizon_is_decoupled_from_segment_stop(monkeypatch):
    monkeypatch.setenv("SKYRL_SCHEDULER_TOTAL_TRAINING_STEPS", "50")
    assert resolve_scheduler_total_training_steps(10) == 50
    monkeypatch.delenv("SKYRL_SCHEDULER_TOTAL_TRAINING_STEPS")
    assert resolve_scheduler_total_training_steps(10) == 10
