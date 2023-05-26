"""Modes

"""

import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp


def _waveguide_operator(omega, epsilon, dx, dy):
  """Waveguide operator as a sparse scipy matrix."""
  _, xx, yy = epsilon.shape

  # Note: We assume row-major -- `flatindex = y + yy * x`.
  dx0 = sp.diags(
      [np.repeat(u, yy) for u in (dx[:, 0], -dx[1:, 0], -dx[0, 0])],
      [i * yy for i in (0, -1, xx - 1)])
  dx1 = sp.diags(
      [np.repeat(u, yy) for u in (dx[:, 1], -dx[:-1, 1], -dx[-1, 1])],
      [i * yy for i in (0, 1, -(xx - 1))])
  dy0 = sp.block_diag(
      [sp.diags((dy[:, 0], -dy[1:, 0], -dy[0, 0]), (0, -1, yy - 1))] * xx)
  dy1 = sp.block_diag(
      [sp.diags((dy[:, 1], -dy[:-1, 1], -dy[-1, 1]), (0, 1, -(yy - 1)))] * xx)

  return (omega**2 * sp.diags(epsilon.ravel()) +
          # Missing some epsilon stuff here.
          sp.vstack([dx0, dy0]) * sp.hstack([dx1, dy1]) +
          sp.vstack([-dy1, dx1]) * sp.hstack([-dy0, dx0]))
