import pytest
import torch
import cached_conv as cc

hparams_list = [{
    "in_channels": 16,
    "out_channels": 16,
    "kernel_size": 3,
    "padding": cc.get_padding(3)
}, {
    "in_channels": 16,
    "out_channels": 16,
    "kernel_size": 5,
    "padding": cc.get_padding(5)
}, {
    "in_channels": 16,
    "out_channels": 16,
    "kernel_size": 3,
    "stride": 2,
    "padding": cc.get_padding(3, 2)
}, {
    "in_channels": 16,
    "out_channels": 16,
    "kernel_size": 7,
    "stride": 2,
    "padding": cc.get_padding(7, 2)
}, {
    "in_channels": 16,
    "out_channels": 16,
    "kernel_size": 7,
    "stride": 4,
    "padding": cc.get_padding(7, 4)
}]


@pytest.mark.parametrize("hparams", hparams_list)
def test_conv(hparams):

    conv = cc.Conv1d(**hparams)
    cconv = cc.CachedConv1d(**hparams)
    cconv.weight.data.copy_(conv.weight.data)
    cconv.bias.data.copy_(conv.bias.data)

    x = torch.randn(1, hparams["in_channels"], 2**14)
    y = conv(x)[..., :-cconv.cumulative_delay]

    cy = cc.chunk_process(cconv, x, 4)[..., cconv.cumulative_delay:]

    assert torch.allclose(y, cy, 1e-5, 1e-5)
