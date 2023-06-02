"""Back out frequency components."""

import jax.numpy as jnp
import numpy as np


def frequency_component(out, steps, omega, dt):
  """Returns E-field at `omega` for simulation output `out` at `steps`."""
  theta = omega * dt * steps
  phases = np.stack([np.cos(theta), -np.sin(theta)], axis=-1)
  parts = jnp.einsum('ij,jk...->ik...', np.linalg.pinv(phases), out)
  return parts[0] + 1j * parts[1]


def source_amplitude(source_waveform, omega, dt):
  """Returns complex scalar denoting source amplitude at `omega`."""
  # `0.5` offset accounts for the E-field source being colocated in time with
  # the H-field updates.
  theta = omega * dt * (jnp.arange(source_waveform.shape[0]) - 0.5)

  # Extracts the complex amplitude of the current source at `omega`.
  parts = jnp.mean((2 *
                    jnp.stack([jnp.cos(theta), -jnp.sin(theta)])[..., None] *
                    source_waveform),
                   axis=1)
  return parts[0] + 1j * parts[1]
