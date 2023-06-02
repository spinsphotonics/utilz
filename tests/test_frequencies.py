from utilz import frequencies

import jax.numpy as jnp
import numpy as np
import pytest


def test_frequency_component():
  omega = 2 * jnp.pi / 40
  steps = jnp.array([10, 20])
  dt = 0.5
  out = frequencies.frequency_component(
      jnp.sin(omega * dt * steps)[:, None, None, None, None], steps, omega, dt)
  print(out)
  assert out[0] == pytest.approx(-1j)


def test_source_amplitude():
  omega = 2 * jnp.pi / 40
  dt = 0.5
  tt = 10000
  out = frequencies.source_amplitude(
      jnp.sin(omega * dt * (jnp.arange(tt) - 0.5))[:, None], omega, dt)
  assert out[0] == pytest.approx(-1j)
