# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test `torch.distributions.Multinomial` resolves through the shape stubs."""

from typing import assert_type, TYPE_CHECKING

from torch.distributions import Multinomial

if TYPE_CHECKING:
    from torch import Tensor


def test_multinomial_construct_and_sample(probs: Tensor[[4, 10]]):
    dist = Multinomial(5000, probs=probs)
    assert_type(dist.total_count, int)
    # Gradual: see the comment on the stub.
    assert_type(dist.sample(), Tensor)
    assert_type(Multinomial(logits=probs, validate_args=False).log_prob(probs), Tensor)
