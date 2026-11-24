"""Build a double-support stopping reference from the measured stance."""

import mujoco
import numpy as np
import torch
from mjlab.utils.lab_api.math import quat_apply, yaw_quat

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

  def launch(
    self,
    initial: ReferenceWindow,
    command: torch.Tensor,
    duration: float,
    foot_speed: torch.Tensor,
    lean: float = 0.0,
    forward_lead: float = 0.0,
    swing_side: int | None = None,
  ) -> ReferenceWindow | None:
    """Shift weight and step in the requested direction from measured support."""
    if not bool(initial.contact.all()) or bool((foot_speed > 0.25).any()):
      return None
    q0, p0, quat0, anchors = (
      value[0].cpu().double().numpy()
      for value in (initial.q, initial.p, initial.quat, initial.feet)
    )
    swing = int(float(command[1]) < 0.0) if swing_side is None else swing_side
    support = 1 - swing
    local = torch.cat((command[:2] * (1.10 * duration), command.new_zeros(1)))
    world = quat_apply(yaw_quat(initial.quat[0]), local).cpu().double().numpy()
    self._update(q0, p0, quat0)
    foot_rotations = self.data.site_xmat[self.sites].reshape(2, 3, 3).copy()
    count = round(duration / self.dt) + 1
    clock = np.linspace(0, 1, count)
    phase = np.clip((clock - 0.23) / 0.77, 0, 1)
    travel = phase**3 * (10 - 15 * phase + 6 * phase**2)
    shift = np.sin(np.pi * clock) ** 2
    feet = np.broadcast_to(anchors, (count, 2, 3)).copy()
    feet[:, swing] += travel[:, None] * world
    feet[:, swing, 2] += 0.055 * np.sin(np.pi * phase) ** 2
    p = p0 + (0.5 * travel[:, None]) * world
    if forward_lead:
      forward = (
        quat_apply(yaw_quat(initial.quat[0]), command.new_tensor((1.0, 0.0, 0.0)))
        .cpu()
        .double()
        .numpy()
      )
      load = np.clip(clock / 0.39, 0, 1)
      load = load**3 * (10 - 15 * load + 6 * load**2)
      p += forward_lead * (load - travel)[:, None] * forward
    toward = anchors[support, :2] - anchors[:, :2].mean(0)
    p[:, :2] += 0.30 * shift[:, None] * toward
    p[:, 2] -= 0.018 * shift
    quat = np.broadcast_to(quat0, (count, 4)).copy()
    if lean:
      axis = np.array([-float(command[1]), float(command[0]), 0.0])
      axis /= np.linalg.norm(axis)
      rotation = np.zeros(4)
      for n, amount in enumerate(shift):
        mujoco.mju_axisAngle2Quat(rotation, axis, lean * amount)
        mujoco.mju_mulQuat(quat[n], quat0, rotation)
    q = q0 + (0.34 * travel[:, None]) * (self.standing - q0)
    for n in range(1, count):
      if not self._project(q[n], p[n], quat[n], feet[n], foot_rotations):
        return None
    angles = initial.q.new_tensor(q)
    velocity = torch.gradient(angles, spacing=self.dt, dim=0)[0]
    velocity[0] = initial.qd[0]
    velocity[-1] = 0.0
    contact = initial.contact[:1].expand(count, -1).clone()
    contact[:, swing] = torch.as_tensor(
      (phase <= 0.0) | (phase >= 1.0), device=contact.device
    )
    return ReferenceWindow(
      angles,
      velocity,
      initial.p.new_tensor(p),
      initial.quat.new_tensor(quat),
      initial.feet.new_tensor(feet),
      contact,
    )

  def capture_stop(
    self,
    initial: ReferenceWindow,
    root_velocity: torch.Tensor,
    foot_velocity: torch.Tensor,
    max_foot_speed: float,
  ) -> ReferenceWindow | None:
    """Add one settling step when only the measured stance foot can be anchored."""
    if int(initial.contact.sum()) != 1:
      return None
    support = int(initial.contact[0, 1])
    swing = 1 - support
    if float(foot_velocity[support, :2].norm()) > max_foot_speed:
      return None

    q0, p0, quat0, anchors = (
      value[0].cpu().double().numpy()
      for value in (initial.q, initial.p, initial.quat, initial.feet)
    )
    root_v = root_velocity.cpu().double().numpy()
    swing_v = foot_velocity[swing].cpu().double().numpy().clip(-0.8, 0.8)
    self._update(q0, p0, quat0)
    foot_rotations = self.data.site_xmat[self.sites].reshape(2, 3, 3).copy()
    upright = yaw_quat(initial.quat)[0].cpu().double().numpy()
    if np.dot(upright, quat0) < 0:
      upright = -upright
    frame = np.empty(9)
    mujoco.mju_quat2Mat(frame, upright)
    axes = frame.reshape(3, 3)[:2, :2]
    omega = np.sqrt(9.81 / np.clip(p0[2] - anchors[support, 2], 0.55, 0.85))
    capture = p0[:2] + root_v[:2] / omega
    forward, side = axes.T @ (capture - anchors[support, :2])
    outward = (-1) ** swing
    landing = anchors[support].copy()
    landing[:2] += axes @ np.array(
      [
        np.clip(forward, -0.12, 0.24),
        outward * np.clip(outward * side + 0.10, 0.15, 0.29),
      ]
    )
    span = max(3, round(0.30 / self.dt))
    duration = span * self.dt
    t = np.linspace(0.0, 1.0, span + 1)
    ease = t**3 * (10 - 15 * t + 6 * t**2)
    tangent = t - 6 * t**3 + 8 * t**4 - 3 * t**5
    n = span + self.count
    feet = np.broadcast_to(anchors, (n, 2, 3)).copy()
    feet[: span + 1, swing] += (
      ease[:, None] * (landing - anchors[swing]) + duration * tangent[:, None] * swing_v
    )
    feet[: span + 1, swing, 2] = (
      np.maximum(feet[: span + 1, swing, 2], anchors[support, 2])
      + 0.025 * np.sin(np.pi * t) ** 2
    )
    feet[0] = anchors
    feet[span + 1 :, swing] = landing

    halfway = p0.copy()
    halfway[:2] += np.clip(root_v[:2] * duration * 0.55, -0.10, 0.10)
    halfway[2] = min(p0[2], 0.79)
    p = np.empty((n, 3))
    p[: span + 1] = p0 + ease[:, None] * (halfway - p0)
    p[: span + 1, :2] += duration * tangent[:, None] * root_v[:2].clip(-0.5, 0.5)
    finish = halfway.copy()
    finish[:2] = 0.5 * (landing[:2] + anchors[support, :2])
    finish[2] = min(halfway[2], 0.78)
    u = np.linspace(0, 1, self.count)
    settle = u**3 * (10 - 15 * u + 6 * u**2)
    p[span:] = halfway + settle[:, None] * (finish - halfway)

    u = np.linspace(0, 1, n)
    blend = u**3 * (10 - 15 * u + 6 * u**2)
    q = q0 + blend[:, None] * (self.standing - q0)
    quat = quat0 + blend[:, None] * (upright - quat0)
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    # Land the free foot at the same roll and pitch as the measured support foot.
    swing_rotation, landing_rotation = np.empty(4), np.empty(4)
    mujoco.mju_mat2Quat(swing_rotation, foot_rotations[swing].reshape(9))
    mujoco.mju_mat2Quat(landing_rotation, foot_rotations[support].reshape(9))
    if np.dot(swing_rotation, landing_rotation) < 0.0:
      landing_rotation = -landing_rotation
    for index in range(1, n):
      a = ease[min(index, span)]
      foot_quat = (1 - a) * swing_rotation + a * landing_rotation
      foot_quat /= np.linalg.norm(foot_quat)
      mujoco.mju_quat2Mat(frame, foot_quat)
      foot_rotations[swing] = frame.reshape(3, 3)
      if not self._project(
        q[index], p[index], quat[index], feet[index], foot_rotations
      ):
        return None
    angles = initial.q.new_tensor(q)
    vel = torch.gradient(angles, spacing=self.dt, dim=0)[0]
    vel[0], vel[-1] = initial.qd[0], 0.0
    contact = initial.contact.expand(n, -1).clone()
    contact[span:, swing] = True
    return ReferenceWindow(
      angles,
      vel,
      initial.p.new_tensor(p),
      initial.quat.new_tensor(quat),
      initial.feet.new_tensor(feet),
      contact,
    )

  def launch_candidates(
    self,
    initial: ReferenceWindow,
    command: torch.Tensor,
    duration: float,
    foot_speed: torch.Tensor,
    lean: float,
    with_alternatives: bool,
    forward_lead: float = 0.0,
  ) -> list[tuple[str, ReferenceWindow]]:
    timings = [(duration, lean)]
    if with_alternatives:
      timings.extend(
        (span, tilt)
        for span in (duration, min(0.8, duration + 0.12))
        for tilt in (0.0, 0.06, 0.12)
        if (span, tilt) != (duration, lean)
      )
    result = []
    axial = abs(float(command[1])) <= 0.10
    choices = (0, 1) if axial else (None,)
    for span, tilt in dict.fromkeys(timings):
      for side in choices:
        window = self.launch(
          initial, command, span, foot_speed, tilt, forward_lead, side
        )
        if window is not None:
          suffix = "" if side is None else ("-left" if side == 0 else "-right")
          result.append((f"contact-launch-{span:.2f}-{tilt:.2f}{suffix}", window))
    return result

  def project_support_seam(
    self,
    plan: ReferenceWindow,
    measured: ReferenceWindow,
    foot_speed: torch.Tensor,
    steps: int,
  ) -> ReferenceWindow:
    """Preserve low-speed measured support while joining a new reference."""
    locked = (measured.contact[0] & (foot_speed <= 0.25)).cpu().numpy().copy()
    if not locked.any() or len(plan.q) <= steps:
      return plan
    q = plan.q[: steps + 1].cpu().double().numpy().copy()
    p = plan.p[: steps + 1].cpu().double().numpy()
    quat = plan.quat[: steps + 1].cpu().double().numpy()
    contacts = plan.contact[: steps + 1].cpu().numpy()
    expiry = np.where((~contacts).any(0), (~contacts).argmax(0), steps)
    anchors = measured.feet[0].cpu().double().numpy()
    self._update(q[0], p[0], quat[0])
    anchor_rotation = self.data.site_xmat[self.sites].reshape(2, 3, 3).copy()
    adjusted_feet = plan.feet[: steps + 1].cpu().double().numpy().copy()
    for i in range(1, steps):
      active = locked & (i < expiry)
      if not active.any():
        break
      self._update(q[i], p[i], quat[i])
      actual = self.data.site_xpos[self.sites].copy()
      rotation = self.data.site_xmat[self.sites].reshape(2, 3, 3).copy()
      decay = np.clip((i / expiry[active] - 0.35) / 0.65, 0.0, 1.0)
      weight = 1.0 - decay**3 * (10 - 15 * decay + 6 * decay**2)
      actual[active] += weight[:, None] * (anchors[active] - actual[active])
      for foot, amount in zip(np.flatnonzero(active), weight, strict=True):
        current, initial = np.empty(4), np.empty(4)
        mujoco.mju_mat2Quat(current, rotation[foot].reshape(9))
        mujoco.mju_mat2Quat(initial, anchor_rotation[foot].reshape(9))
        if np.dot(current, initial) < 0.0:
          initial = -initial
        blended = (1 - amount) * current + amount * initial
        mujoco.mju_quat2Mat(
          rotation[foot].reshape(9), blended / np.linalg.norm(blended)
        )
      if not self._project(q[i], p[i], quat[i], actual, rotation):
        return plan
      adjusted_feet[i] = actual
    angles = plan.q.new_tensor(q)
    delta = angles - plan.q[: steps + 1]
    if not bool(delta.any()):
      return plan
    correction_rate = torch.gradient(delta, spacing=self.dt, dim=0)[0]
    correction_rate[0] = correction_rate[-1] = 0.0
    return ReferenceWindow(
      torch.cat((angles, plan.q[steps + 1 :])),
      torch.cat((plan.qd[: steps + 1] + correction_rate, plan.qd[steps + 1 :])),
      plan.p,
      plan.quat,
      torch.cat((plan.feet.new_tensor(adjusted_feet), plan.feet[steps + 1 :])),
      plan.contact,
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
