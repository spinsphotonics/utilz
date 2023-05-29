"""Modes for finite-difference Yee-cell grids."""

import numpy as np
import scipy.sparse as sp


def _waveguide_operator(omega, epsilon, dx, dy):
  """Waveguide operator as a sparse scipy matrix."""
  # Note: We assume row-major -- `flatindex = y + yy * x`.
  _, xx, yy = epsilon.shape
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
  exy = sp.diags(epsilon[:2].ravel())
  inv_ez = sp.diags(1 / epsilon[2].ravel())
  return (omega**2 * exy -
          sp.vstack([dx0, dy0]) * inv_ez * sp.hstack([dx1, dy1]) * exy -
          sp.vstack([-dy1, dx1]) * sp.hstack([-dy0, dx0]))


def _find_largest_eigenvalue(A, numsteps):
  """Estimate dominant eigenvector using power iteration."""
  v = np.random.rand(A.shape[0])
  for _ in range(numsteps):
    v = A @ v
    v /= np.linalg.norm(v)
  return v @ A @ v


def waveguide(i, omega, epsilon, dx, dy):
  """Solves for the `i`th mode of the waveguide at `omega`.

  Assumes a real-valued structure in the x-y plane and propagation along the
  z-axis according to `exp(-i * wavevector* z)`. Uses dimensionless units and
  periodic boundaries.

  Currently does not use JAX and is not differentiable -- waiting on a
  sparse eigenvalue solver in JAX.

  Args:
    i: Mode number to solve for, where `i = 0` corresponds to the fundamental
      mode of the structure.
    omega: Angular frequency of the mode.
    epsilon: `(3, xx, yy)` array of permittivity values for Ex, Ey, and Ez
      nodes on a finite-difference Yee grid.
    dx: `(xx, 2)` array of cell sizes in the x-direction. `[:, 0]` values
      correspond to Ey/Ez components while `[:, 1]` values correspond to
      Ex components.
    dy: `(yy, 2)` array similar to `dx` but for cell sizes along `y`.

  Returns:
    wavevector: Real-valued scalar.
    fields: `(2, xx, yy)` array of real-valued Ex and Ey field values of the
      mode.

  """
  A = _waveguide_operator(omega, epsilon, dx, dy)
  shift = _find_largest_eigenvalue(A, 20)
  if shift >= 0:
    raise ValueError("Expected largest eigenvalue to be negative.")
  w, v = sp.linalg.eigs(A - shift * sp.eye(A.shape[0]), k=i+1, which="LM")
  beta = np.real(np.sqrt(w[i] + shift))
  mode = np.reshape(np.real(v[:, i]), (2,) + epsilon.shape[1:])
  if beta == 0:
      raise ValueError("No propagating mode found.")
  return np.float32(beta),  np.float32(mode)
