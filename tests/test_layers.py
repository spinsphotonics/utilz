from utilz import layers

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("domain,m,vals", [((4, 5, 6), 1, (1, 3))])
@pytest.mark.parametrize("thresh,index,expected", [
    # x-threshold.
    pytest.param((5, 0, -0.5), (0, 1, 0, 0), 1.0, id="x-Ex-pre"),
    pytest.param((5, 0, -0.5), (0, 2, 0, 0), 1.5, id="x-Ex-on"),
    pytest.param((5, 0, -0.5), (0, 3, 0, 0), 3.0, id="x-Ex-post"),
    pytest.param((4, 0, -0.5), (1, 1, 0, 0), 1.0, id="x-Ey-pre"),
    pytest.param((4, 0, -0.5), (1, 2, 0, 0), 2.0, id="x-Ey-on"),
    pytest.param((4, 0, -0.5), (1, 3, 0, 0), 3.0, id="x-Ey-post"),
])
def test_simple(domain, m, vals, thresh, index, expected):
  xx, yy, zz = domain
  layer = vals[0] * jnp.ones((2, 2 * m * xx, 2 * m * yy))
  layer = layer.at[1, thresh[0]:, thresh[1]:].set(
      vals[1])
  out = layers.render(layer,
                      jnp.array([thresh[2]]),
                      jnp.arange(zz)[:, None] + jnp.array([[-0.5, 0]]),
                      jnp.arange(zz)[:, None] + jnp.array([[0.5, 1]]),
                      m)
  assert out[index] == pytest.approx(expected)