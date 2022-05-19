import pytest
import torch
import cached_conv as cc
import torch.nn as nn

hparams_list = [
    {
        "in_channels": 16,
        "out_channels": 16,
        "kernel_size": 4,
        "stride": 2,
        "padding": 1
    },
    {
        "in_channels": 16,
        "out_channels": 16,
        "kernel_size": 8,
        "stride": 4,
        "padding": 2
    },
    {
        "in_channels": 16,
        "out_channels": 16,
        "kernel_size": 12,
        "stride": 4,
        "padding": 4
    },
]


@pytest.mark.parametrize("hparams", hparams_list)
def test_conv_t(hparams):
    model_constructor = lambda: cc.ConvTranspose1d(**hparams)
    input_tensor = torch.randn(1, hparams["out_channels"], 2**14)
    assert cc.test_equal(model_constructor, input_tensor)
