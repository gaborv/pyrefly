# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test `F.one_hot`, which appends a class dimension."""

from typing import assert_type, TYPE_CHECKING

import torch.nn.functional as F
from shape_extensions import IntVar

if TYPE_CHECKING:
    from torch import Tensor


def test_one_hot_literal_classes(x: Tensor[[2, 3]]):
    assert_type(F.one_hot(x, 7), Tensor[[2, 3, 7]])


def test_one_hot_symbolic_batch[B: IntVar](x: Tensor[[B]]):
    assert_type(F.one_hot(x, 4), Tensor[[B, 4]])


def test_one_hot_runtime_classes(x: Tensor[[2, 3]], n: int):
    assert_type(F.one_hot(x, n), Tensor[[2, 3, int]])
    assert_type(F.one_hot(x), Tensor[[2, 3, int]])
