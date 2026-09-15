# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test `torch.empty` with a runtime size sequence, matching `zeros` and `ones`."""

from typing import assert_type, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor


def test_empty_with_list_size(size: list[int], n: int, x: Tensor[[2, 3]]):
    assert_type(torch.empty(size), Tensor)
    assert_type(
        torch.empty([n, x.shape[1]], dtype=torch.float32, requires_grad=True), Tensor
    )
    assert_type(torch.zeros([n, x.shape[1]]), Tensor)
