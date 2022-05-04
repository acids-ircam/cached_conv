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
    cc.use_cached_conv(False)
    conv = cc.ConvTranspose1d(**hparams)

    cc.use_cached_conv(True)
    cconv = cc.ConvTranspose1d(**hparams)

    cconv.weight.data.copy_(conv.weight.data)
    cconv.bias.data.copy_(conv.bias.data)

    x = torch.randn(1, hparams["out_channels"], 2**14)

    y = conv(x)[..., :-cconv.cumulative_delay]
    cy = cc.chunk_process(cconv, x, 4)[..., cconv.cumulative_delay:]

    assert torch.allclose(y, cy, 1e-5, 1e-5)
