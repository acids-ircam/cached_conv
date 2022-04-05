import pytest
import cached_conv as cc
import torch
import matplotlib.pyplot as plt

hparams_list = [{
    "dim": 1,
    "kernels": [3, 5, 7, 9],
    "strides": [1, 1, 1, 1]
}, {
    "dim": 1,
    "kernels": [3, 5, 7, 9],
    "strides": [2, 1, 1, 1]
}, {
    "dim": 1,
    "kernels": [3, 5, 7, 9],
    "strides": [1, 2, 1, 1]
}, {
    "dim": 1,
    "kernels": [3, 5, 7, 9],
    "strides": [1, 1, 2, 1]
}, {
    "dim": 1,
    "kernels": [3, 3, 3, 3],
    "strides": [2, 1, 2, 1]
}]


def build_models(dim, kernels, strides):
    layers = []
    clayers = []

    cum_delay = 0
    for k, s in zip(kernels, strides):
        layers.append(
            cc.Conv1d(
                dim,
                dim,
                k,
                stride=s,
                padding=cc.get_padding(k, s),
                cumulative_delay=cum_delay,
            ))
        clayers.append(
            cc.CachedConv1d(
                dim,
                dim,
                k,
                stride=s,
                padding=cc.get_padding(k, s),
                cumulative_delay=cum_delay,
            ))
        cum_delay = clayers[-1].cumulative_delay

    model = cc.CachedSequential(*layers)
    cmodel = cc.CachedSequential(*clayers)

    for i in range(len(model)):
        model[i].weight.data.copy_(cmodel[i].weight.data)
        model[i].bias.data.copy_(cmodel[i].bias.data)

    return model, cmodel


@pytest.mark.parametrize("hparams", hparams_list)
def test_sequential(hparams):
    x = torch.randn(1, hparams["dim"], 2**14)
    model, cmodel = build_models(**hparams)

    fc = cmodel.cumulative_delay

    y = model(x)[..., :-fc]
    cy = cc.chunk_process(cmodel, x, 16)[..., fc:]

    assert torch.allclose(y[..., fc:-fc], cy[..., fc:-fc], 1e-5, 1e-5)
