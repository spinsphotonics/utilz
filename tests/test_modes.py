from utilz import modes

import numpy as np
import pytest


def test_float32():
  xx, yy = 30, 20
  epsilon = np.ones((3, xx, yy))
  epsilon[:, 9:21, 8:12] = 12.25
  beta, field = modes.waveguide(0, 2 * np.pi / 37, epsilon,
                                np.ones((xx, 2)), np.ones((yy, 2)))
  assert beta.dtype == np.float32
  assert field.dtype == np.float32


@pytest.mark.parametrize("i,expected", [
    (0, 0.36388508),
    (1, 0.18891069),
    (2, 0.15406249),
    (3, 0.13549446),
])
def test_find_modes(i, expected):
  xx, yy = 30, 20
  epsilon = np.ones((3, xx, yy))
  epsilon[:, 9:21, 8:12] = 12.25
  beta, field = modes.waveguide(i, 2 * np.pi / 37, epsilon,
                                np.ones((xx, 2)), np.ones((yy, 2)))
  assert beta == pytest.approx(expected)


def test_no_propagating_mode():
  xx, yy = 30, 20
  epsilon = np.ones((3, xx, yy))
  epsilon[:, 9:21, 8:12] = 12.25
  with pytest.raises(ValueError, match="No propagating mode found"):
    beta, field = modes.waveguide(4, 2 * np.pi / 37, epsilon,
                                  np.ones((xx, 2)), np.ones((yy, 2)))
