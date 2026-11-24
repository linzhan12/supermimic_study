"""Build a double-support stopping reference from the measured stance."""

import mujoco
import numpy as np
import torch
from mjlab.utils.lab_api.math import yaw_quat

from supermimic.locomotion_planner import ReferenceWindow


class StanceBridge:
  """Project a settling motion onto two fixed foot sites on a scratch MjData."""

  def __init__(
    self,
    model: mujoco.MjModel,
    root: torch.Tensor,
    joints: torch.Tensor,
    legs: tuple[int, ...],
    sites: torch.Tensor,
    standing: torch.Tensor,
    dt: float,
    duration: float,
  ) -> None:
    self.model = model
    self.data = mujoco.MjData(model)
    self.root = root.cpu().numpy()
    self.joints = joints.cpu().numpy()
    self.legs = np.asarray(legs, dtype=int)
    self.sites = sites.cpu().numpy()
    joint_ids = np.asarray(
      [np.flatnonzero(model.jnt_qposadr == i).item() for i in self.joints[self.legs]]
    )
    self.dofs = model.jnt_dofadr[joint_ids]
    self.limits = model.jnt_range[joint_ids]
    self.standing = standing.cpu().double().numpy()
    self.dt = dt
    self.count = max(3, round(duration / dt) + 1)
    self.jacp = np.zeros((3, model.nv))
    self.jacr = np.zeros_like(self.jacp)

  def __call__(
    self,
    initial: ReferenceWindow,
    foot_speed: torch.Tensor,
    max_foot_speed: float | None,
  ) -> ReferenceWindow | None:
    """Keep the measured contact positions throughout the settling motion."""
    if not bool(initial.contact[0].all()):
      return None
    if max_foot_speed is not None and bool((foot_speed > max_foot_speed).any()):
      return None
    q0, p0, quat0, anchors = (
      value[0].cpu().double().numpy()
      for value in (initial.q, initial.p, initial.quat, initial.feet)
    )
    self._update(q0, p0, quat0)
    foot_rotations = self.data.site_xmat[self.sites].reshape(2, 3, 3).copy()
    upright = yaw_quat(initial.quat)[0].cpu().double().numpy()
    if np.dot(upright, quat0) < 0:
      upright = -upright
    center = p0.copy()
    center[:2] = anchors[:, :2].mean(axis=0)
    center[2] = min(p0[2], 0.78)
    time = np.linspace(0.0, 1.0, self.count)[:, None]
    blend = time**3 * (10 - 15 * time + 6 * time**2)
    q = q0 + blend * (self.standing - q0)
    p = p0 + blend * (center - p0)
    quat = quat0 + blend * (upright - quat0)
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    for n in range(1, self.count):
      if not self._project(q[n], p[n], quat[n], anchors, foot_rotations):
        return None
    angles = initial.q.new_tensor(q)
    joint_velocity = torch.gradient(angles, spacing=self.dt, dim=0)[0]
    joint_velocity[0] = initial.qd[0]
    joint_velocity[-1] = 0.0
    return ReferenceWindow(
      angles,
      joint_velocity,
      initial.p.new_tensor(p),
      initial.quat.new_tensor(quat),
      initial.feet[:1].expand(self.count, -1, -1),
      initial.contact[:1].expand(self.count, -1),
    )

  def _update(self, q: np.ndarray, p: np.ndarray, quat: np.ndarray) -> None:
    self.data.qpos[self.joints] = q
    self.data.qpos[self.root[:3]] = p
    self.data.qpos[self.root[3:]] = quat
    mujoco.mj_kinematics(self.model, self.data)
    mujoco.mj_comPos(self.model, self.data)

  def _project(
    self,
    q: np.ndarray,
    p: np.ndarray,
    quat: np.ndarray,
    anchors: np.ndarray,
    foot_rotations: np.ndarray,
  ) -> bool:
    rotation_weight = 0.08
    damping = 0.002
    identity = np.eye(len(self.dofs))
    for _ in range(12):
      self._update(q, p, quat)
      pos_error = anchors - self.data.site_xpos[self.sites]
      rotations = self.data.site_xmat[self.sites].reshape(2, 3, 3)
      angle_error = 0.5 * np.cross(rotations, foot_rotations, axis=1).sum(axis=2)
      if (
        np.max(np.linalg.norm(pos_error, axis=1)) < 0.004
        and np.max(abs(angle_error)) < 0.06
      ):
        return True
      jac = []
      for site in self.sites:
        mujoco.mj_jacSite(self.model, self.data, self.jacp, self.jacr, int(site))
        jac.append(
          np.vstack(
            (self.jacp[:, self.dofs], rotation_weight * self.jacr[:, self.dofs])
          )
        )
      matrix = np.vstack(jac)
      error = np.concatenate(
        (pos_error, rotation_weight * angle_error), axis=1
      ).reshape(-1)
      delta = np.linalg.solve(
        matrix.T @ matrix + damping**2 * identity, matrix.T @ error
      )
      q[self.legs] = np.clip(q[self.legs] + np.clip(delta, -0.12, 0.12), *self.limits.T)
    return False
