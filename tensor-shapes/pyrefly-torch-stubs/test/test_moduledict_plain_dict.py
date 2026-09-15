# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test nn.ModuleDict built from a plain dict, without a TypedDict."""

from typing import assert_type, TYPE_CHECKING

import torch.nn as nn

if TYPE_CHECKING:
    from torch import Tensor


class Heads(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.ModuleDict({"a": nn.Linear(8, 4), "b": nn.Linear(8, 2)})

    def forward(self, x: Tensor) -> Tensor:
        return self.proj["a"](x)


def test_plain_dict_entries_are_modules(heads: Heads, name: str) -> None:
    assert_type(heads.proj["a"], nn.Module)
    assert_type(heads.proj[name], nn.Module)
    for key, module in heads.proj.items():
        assert_type(key, str)
        assert_type(module, nn.Module)
    for module in heads.proj.values():
        assert_type(module, nn.Module)
