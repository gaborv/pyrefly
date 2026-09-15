# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test `Normal` with Python scalar parameters, which PyTorch accepts and broadcasts."""

from typing import assert_type, TYPE_CHECKING

from torch.distributions import Normal

if TYPE_CHECKING:
    from torch import Tensor


def test_normal_tensor_params_keep_event_shape(
    loc: Tensor[[2, 3]], scale: Tensor[[2, 3]]
):
    assert_type(Normal(loc, scale).rsample(), Tensor[[2, 3]])


def test_normal_one_scalar_param_keeps_tensor_shape(t: Tensor[[2, 3]], mu: float):
    assert_type(Normal(t, 1.0).rsample(), Tensor[[2, 3]])
    assert_type(Normal(mu, t).rsample(), Tensor[[2, 3]])


def test_normal_scalar_params_stay_gradual(x: Tensor[[5]], mu: float, sigma: float):
    # `log_prob` returns the batch shape, which is `()` here, while PyTorch
    # broadcasts it with `value`; a gradual result avoids a wrong `Tensor[[]]`.
    assert_type(Normal(mu, sigma).log_prob(x), Tensor)
    assert_type(Normal(loc=0.0, scale=1.0, validate_args=False).sample(), Tensor)
