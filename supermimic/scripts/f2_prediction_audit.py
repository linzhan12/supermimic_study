"""Paired predictions and physical transitions for the evaluated F2 controller."""

import json
from pathlib import Path

import torch

from supermimic.contact_tracking import ContactTrackingAdapter
from supermimic.locomotion_planner import (
  PREVIEW_STEPS,
  LocomotionPlanner,
  ReferenceWindow,
)
from supermimic.reference_trackability import projected_twist, tracking_steps
from supermimic.runners.supermimic_runner import WorldModel

HORIZONS = (1, 5, 10, 20)
WINDOW_FIELDS = ("q", "qd", "p", "quat", "feet", "contact")


class PredictionAudit:
  def __init__(
    self,
    stride: int,
    dt: float,
    heading_strength: float,
    adapter: ContactTrackingAdapter | None,
    adapter_gain: float,
  ) -> None:
    self.stride = stride
    self.dt = dt
    self.heading_strength = heading_strength
    self.adapter = adapter
    self.adapter_gain = adapter_gain
    self.previous_plan = None
    self.revision = -1
    self.steps: list[dict] = []
    self.windows: list[ReferenceWindow] = []
    self.indices: list[int] = []

  def record(
    self,
    planner: LocomotionPlanner,
    phase: str,
    state: torch.Tensor,
    aux: torch.Tensor,
    history: torch.Tensor,
    goal: torch.Tensor,
    action: torch.Tensor,
    command: torch.Tensor,
    contact: torch.Tensor,
    next_state: torch.Tensor,
    next_aux: torch.Tensor,
    next_contact: torch.Tensor,
    adapted: bool,
    heading_anchor: torch.Tensor | None,
  ) -> None:
    if self.previous_plan is not planner.plan:
      self.previous_plan = planner.plan
      self.revision += 1
    if len(self.steps) % self.stride == 0:
      self.indices.append(len(self.steps))
      self.windows.append(
        planner.plan.hold_after(planner.offset, PREVIEW_STEPS + HORIZONS[-1] - 1)
      )
    row = {
      "state": state[0],
      "aux": aux[0],
      "history": history[0],
      "goal": goal[0],
      "action": action[0],
      "command": command,
      "contact": contact[0],
      "next_state": next_state[0],
      "next_aux": next_aux[0],
      "next_contact": next_contact[0],
      "heading_anchor": aux[0, 3:] if heading_anchor is None else heading_anchor[0],
      "heading_lock": torch.tensor(heading_anchor is not None),
    }
    self.steps.append(
      {
        **{key: value.detach().cpu().clone() for key, value in row.items()},
        "phase": phase,
        "reference_revision": self.revision,
        "adapted": adapted,
      }
    )

  def write(self, path: Path, wm: WorldModel, actor, context: torch.Tensor) -> None:
    if not self.steps:
      return
    device = context.device
    fields = {
      name: torch.stack([row[name] for row in self.steps])
      for name, value in self.steps[0].items()
      if isinstance(value, torch.Tensor)
    }
    saved_windows = {
      name: torch.stack([getattr(window, name) for window in self.windows]).cpu()
      for name in WINDOW_FIELDS
    }
    torch.save(
      {
        **fields,
        "context": context.cpu(),
        "dt": self.dt,
        "heading_strength": self.heading_strength,
        "window_steps": self.indices,
        "windows": saved_windows,
        "phase": [row["phase"] for row in self.steps],
        "reference_revision": [row["reference_revision"] for row in self.steps],
      },
      path.with_suffix(".pt"),
    )
    rep = wm.representation
    transitions = []
    for start, window in zip(self.indices, self.windows, strict=True):
      steps = min(HORIZONS[-1], len(self.steps) - start)
      index = slice(start, start + steps)
      state = fields["state"][start : start + 1].to(device)
      aux = fields["aux"][start : start + 1].to(device)
      actions = fields["action"][index].to(device)
      actual_aux = fields["next_aux"][index].to(device)
      actual_q = rep.joint_state(fields["next_state"][index].to(device))[
        :, : window.q.shape[-1]
      ]
      cmd = fields["command"][start].to(device)
      history = fields["history"][start : start + 1].to(device)
      same_reference = 0
      for row in self.steps[index]:
        if row["reference_revision"] != self.steps[start][
          "reference_revision"
        ] or not torch.equal(row["command"], self.steps[start]["command"]):
          break
        same_reference += 1
      # Teacher-forced actions isolate the learned dynamics. The independent
      # policy rollout is comparable only until a replanned window takes over.
      physical_actions_aux, physical_actions_q = [], []
      next_aux, next_state = aux, state
      for action in actions:
        next_aux, next_state = wm.dynamics(next_state, action[None], next_aux, context)
        physical_actions_aux.append(next_aux[0])
        physical_actions_q.append(rep.joint_state(next_state)[0, : window.q.shape[-1]])
      receding = list(
        tracking_steps(
          [window],
          actor,
          wm.dynamics,
          rep,
          state,
          aux,
          history,
          context,
          same_reference,
          dt=self.dt,
          yaw_rate=float(cmd[2]) if self.heading_strength else 0.0,
          heading_strength=self.heading_strength,
          heading_anchor=fields["heading_anchor"][start : start + 1].to(device)
          if bool(fields["heading_lock"][start])
          else None,
          adapter=self.adapter if self.steps[start]["adapted"] else None,
          command=cmd,
          measured_contact=fields["contact"][start : start + 1].to(device),
          adapter_gain=self.adapter_gain,
        )
      )
      torch.testing.assert_close(receding[0][2], actions[:1], rtol=1e-5, atol=1e-6)
      for horizon in HORIZONS:
        if horizon > steps:
          break
        end = horizon - 1
        actual = projected_twist(aux, actual_aux[end : end + 1], self.dt * horizon)[0]
        forced = projected_twist(
          aux, physical_actions_aux[end][None], self.dt * horizon
        )[0]
        row = {
          "step": start,
          "horizon": horizon,
          "phase": self.steps[start]["phase"],
          "contact": fields["contact"][start].tolist(),
          "contact_changed": bool(
            (
              fields["next_contact"][start : start + horizon]
              != fields["contact"][start]
            ).any()
          ),
          "actual": actual.tolist(),
          "forced_actions": forced.tolist(),
          "forced_joint_rmse_rad": float(
            (physical_actions_q[end] - actual_q[end]).square().mean().sqrt()
          ),
        }
        if horizon <= same_reference:
          pose, joints, _ = receding[end]
          row["closed_loop"] = projected_twist(aux, pose, self.dt * horizon)[0].tolist()
          row["closed_joint_rmse_rad"] = float(
            (joints[0, : window.q.shape[-1]] - actual_q[end]).square().mean().sqrt()
          )
        transitions.append(row)
    path.write_text(
      "\n".join(json.dumps(r, allow_nan=False) for r in transitions) + "\n"
    )
