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
    model_constructor = lambda: cc.Conv1d(**hparams)
    input_tensor = torch.randn(1, hparams["in_channels"], 2**14)
    assert cc.test_equal(model_constructor, input_tensor)
