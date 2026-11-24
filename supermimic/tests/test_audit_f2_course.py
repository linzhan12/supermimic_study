import math

import pytest

from scripts.audit_f2_course import course_metrics


@pytest.mark.parametrize("drift", [0.0, 0.03])
def test_complete_cycle_preserves_drift_and_cancels_periodic_sway(drift):
  dt = 0.02
  speed = 0.5
  rate = math.tau * 0.9
  rows = []
  for frame in range(125):
    t = frame * dt
    rows.append({
      "phase": "fast",
      "target_velocity": [speed, 0.0, 0.0],
      "pelvis_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
      "pelvis_position_w_m": [speed * t, drift * t + 0.04 * math.sin(rate * t), 0.8],
      "actual_velocity": [speed, drift + 0.04 * rate * math.cos(rate * t), 0.0],
    })
  (result,) = course_metrics(rows, dt)
  assert result["last_cycle_ground_lateral_mps"] == pytest.approx(drift, abs=1e-4)
  assert abs(result["last_half_second_body_lateral_mps"] - drift) > 0.05
