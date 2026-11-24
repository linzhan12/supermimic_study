import numpy as np

from scripts.audit_forward_gait import bridge_single_ground_dropout


def test_bridge_fills_only_an_isolated_near_ground_dropout():
  contact = np.array(
    [[True, True], [False, False], [True, True], [False, True], [False, True], [True, True]]
  )
  soles = np.zeros_like(contact, dtype=float)
  soles[1, 1] = 0.004
  result = bridge_single_ground_dropout(contact, soles)
  expected = contact.copy()
  expected[1, 0] = True
  np.testing.assert_array_equal(result, expected)
  assert not contact[1, 0]
