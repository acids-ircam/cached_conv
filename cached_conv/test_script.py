import torch
import pytest
import cached_conv as cc

torch.set_grad_enabled(False)

model_list = [
    (cc.CachedConv1d, {
        "in_channels": 16,
        "out_channels": 16,
        "kernel_size": 3,
        "padding": cc.get_padding(3)
    }),
    (cc.CachedConvTranspose1d, {
        "in_channels": 16,
        "out_channels": 16,
        "kernel_size": 4,
        "stride": 2,
        "padding": 1
    }),
]


@pytest.mark.parametrize("model", model_list)
def test_script(model):

    model, hp = model[0](**model[1]), model[1]
    x = torch.randn(1, hp["in_channels"], 2**14)
    model(x)
    script = torch.jit.script(model)