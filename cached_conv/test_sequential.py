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


def build_model(dim, kernels, strides):
    layers = []
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
        cum_delay = layers[-1].cumulative_delay

    model = cc.CachedSequential(*layers)
    return model


@pytest.mark.parametrize("hparams", hparams_list)
def test_sequential(hparams):
    model_constructor = lambda: build_model(**hparams)
    input_tensor = torch.randn(1, hparams["dim"], 2**14)
    assert cc.test_equal(model_constructor, input_tensor)
