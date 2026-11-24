from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from supermimic.tasks.forward_gait import recorded_gait_actions


def test_gait_teacher_uses_next_transition_in_the_same_clip(tmp_path: Path):
  names = ("left_knee_joint", "right_knee_joint")
  first, second = (tmp_path / name for name in ("first.npz", "second.npz"))
  actions = np.arange(14, dtype=np.float32).reshape(7, 2)
  np.savez(first, executed_action=actions[:4], action_names=names)
  np.savez(second, executed_action=actions[4:], action_names=names)
  prior = SimpleNamespace(
    reference_files=(str(first), str(second)),
    reference_motion=SimpleNamespace(
      motion_ends=torch.tensor([4, 7]), joint_pos=torch.zeros(7, 2)
    ),
    reference_rows=torch.tensor([0, 2, 4, 5]),
  )

  torch.testing.assert_close(
    recorded_gait_actions(prior, names), torch.from_numpy(actions[[1, 3, 5, 6]])
  )
  prior.reference_rows = torch.tensor([3])
  with pytest.raises(ValueError, match="terminal demonstration state"):
    recorded_gait_actions(prior, names)
  with pytest.raises(ValueError, match="joint order"):
    recorded_gait_actions(prior, names[::-1])


def test_npz_without_recorded_controls_keeps_frozen_actor_teacher(tmp_path: Path):
  path = tmp_path / "legacy.npz"
  np.savez(path, joint_pos=np.zeros((3, 1)))
  prior = SimpleNamespace(
    reference_files=(str(path),),
    reference_motion=SimpleNamespace(motion_ends=torch.tensor([3])),
  )
  assert recorded_gait_actions(prior, ("left_knee_joint",)) is None
