# Copyright 2023 The JAX Authors.
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

import functools
import itertools

from jax._src import core
from jax._src.interpreters import ad
from jax._src.interpreters import mlir
from jax._src.tree_util import tree_flatten, tree_unflatten
from jax._src.util import safe_zip
from jax._src.lib import xla_client as xc
from jax._src.lib.mlir import ir

_next_shard_group_id = itertools.count()

def shard_like(x, y):
  """Shards x and y like each other."""
  x_flat, x_tree = tree_flatten(x)
  y_flat, y_tree = tree_flatten(y)

  if x_tree != y_tree:
    raise ValueError('Trees should be equal')

  outs = [shard_like_p.bind(t, s) for t, s in safe_zip(x_flat, y_flat)]
  x_out_flat, y_out_flat = zip(*outs)
  return tree_unflatten(x_tree, x_out_flat), tree_unflatten(y_tree, y_out_flat)


shard_like_p = core.Primitive('shard_like')
shard_like_p.multiple_results = True

def _shard_like_impl(x, y):
  raise ValueError('shard_like only works inside jax.jit.')
shard_like_p.def_impl(_shard_like_impl)
shard_like_p.def_abstract_eval(lambda x, y: (x, y))

def shard_like_transpose_rule(ct, x, y):
  return [shard_like_p.bind(ct, y, x)]
ad.deflinear2(shard_like_p, shard_like_transpose_rule)


def _group_shard(
    type: xc.OpSharding.ShardGroupType,
    ctx,
    x: ir.Value,
    y: ir.Value,
    x_aval_out: core.AbstractValue,
    y_aval_out: core.AbstractValue,
) -> tuple[ir.Value, ir.Value]:
  shard_group_id = next(_next_shard_group_id)

  unknown_op_sharding = xc.OpSharding()
  unknown_op_sharding.type = xc.OpSharding.Type.UNKNOWN
  unknown_op_sharding.is_shard_group = True
  unknown_op_sharding.shard_group_id = shard_group_id
  unknown_op_sharding.shard_group_type = type

  x = mlir.wrap_with_sharding_op(ctx, x, x_aval_out, unknown_op_sharding)
  y = mlir.wrap_with_sharding_op(ctx, y, y_aval_out, unknown_op_sharding)
  return x, y


mlir_shard_as = functools.partial(_group_shard, xc.OpSharding.ShardGroupType.AS)

def shard_like_lowering(ctx, x, y):
  return mlir_shard_as(ctx, x, y, *ctx.avals_out)
mlir.register_lowering(shard_like_p, shard_like_lowering)
