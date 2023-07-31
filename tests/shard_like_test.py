# Copyright 2021 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from jax._src import xla_bridge
from jax._src import test_util as jtu
from jax.sharding import NamedSharding, PartitionSpec as P
from jax.experimental.shard_like import shard_like

from jax import config
config.parse_flags_with_absl()

prev_xla_flags = None


def setUpModule():
  global prev_xla_flags
  prev_xla_flags = os.getenv("XLA_FLAGS")
  flags_str = prev_xla_flags or ""
  # Don't override user-specified device count, or other XLA flags.
  if "xla_force_host_platform_device_count" not in flags_str:
    os.environ["XLA_FLAGS"] = (flags_str +
                               " --xla_force_host_platform_device_count=8")
  # Clear any cached backends so new CPU backend will pick up the env var.
  xla_bridge.get_backend.cache_clear()

def tearDownModule():
  if prev_xla_flags is None:
    del os.environ["XLA_FLAGS"]
  else:
    os.environ["XLA_FLAGS"] = prev_xla_flags
  xla_bridge.get_backend.cache_clear()


class PJitTest(jtu.JaxTestCase):

  def test_basic_shard_like(self):
    mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    np_inp = np.arange(16).reshape(8, 2)
    inp = jax.device_put(np_inp, NamedSharding(mesh, P('x', 'y')))

    @jax.jit
    def f(x):
      y = x * 2
      z = y @ y.T
      _, z = shard_like(y, z)
      return z * 2

    print(f.lower(inp).as_text('hlo'))
    out = f(inp)
    print(out, out.sharding)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
