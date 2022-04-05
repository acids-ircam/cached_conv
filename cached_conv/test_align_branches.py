import pytest
import cached_conv as cc
import torch
import torch.nn as nn


def test_residual():
    model = cc.Conv1d(1, 1, 5, padding=cc.get_padding(5))
    cmodel = cc.CachedConv1d(1, 1, 5, padding=cc.get_padding(5))

    cmodel.weight.data.copy_(model.weight.data)
    cmodel.bias.data.copy_(model.bias.data)

    model = cc.AlignBranches(
        model,
        nn.Identity(),
        delays=[model.cumulative_delay, 0],
    )

    cmodel = cc.AlignBranches(
        cmodel,
        nn.Identity(),
        delays=[cmodel.cumulative_delay, 0],
    )

    x = torch.randn(1, 1, 2**14)

    y = sum(model(x))[..., :-cmodel.cumulative_delay]
    cy = sum(cmodel(x))[..., cmodel.cumulative_delay:]

    assert torch.allclose(y, cy, 1e-3, 1e-3)


def test_parallel():
    b1 = cc.Conv1d(1, 1, 5, padding=cc.get_padding(5))
    cb1 = cc.CachedConv1d(1, 1, 5, padding=cc.get_padding(5))

    cb1.weight.data.copy_(b1.weight.data)
    cb1.bias.data.copy_(b1.bias.data)

    b2 = cc.Conv1d(1, 1, 3, padding=cc.get_padding(3))
    cb2 = cc.CachedConv1d(1, 1, 3, padding=cc.get_padding(3))

    cb2.weight.data.copy_(b2.weight.data)
    cb2.bias.data.copy_(b2.bias.data)

    model = cc.AlignBranches(b1, b2)
    cmodel = cc.AlignBranches(cb1, cb2)

    x = torch.randn(1, 1, 2**14)

    y = sum(model(x))[..., :-cmodel.cumulative_delay]
    cy = sum(cmodel(x))[..., cmodel.cumulative_delay:]

    assert torch.allclose(y, cy, 1e-3, 1e-3)


def test_parallel_stride():
    b1 = cc.Conv1d(1, 1, 5, stride=2, padding=cc.get_padding(5, 2))
    cb1 = cc.CachedConv1d(1, 1, 5, stride=2, padding=cc.get_padding(5, 2))

    cb1.weight.data.copy_(b1.weight.data)
    cb1.bias.data.copy_(b1.bias.data)

    b2 = cc.Conv1d(1, 1, 3, stride=2, padding=cc.get_padding(3, 2))
    cb2 = cc.CachedConv1d(1, 1, 3, stride=2, padding=cc.get_padding(3, 2))

    cb2.weight.data.copy_(b2.weight.data)
    cb2.bias.data.copy_(b2.bias.data)

    model = cc.AlignBranches(b1, b2, stride=.5)
    cmodel = cc.AlignBranches(cb1, cb2, stride=.5)

    x = torch.randn(1, 1, 2**14)

    y = sum(model(x))[..., :-cmodel.cumulative_delay]
    cy = sum(cmodel(x))[..., cmodel.cumulative_delay:]

    assert torch.allclose(y, cy, 1e-3, 1e-3)


def test_parallel_transpose():
    b1 = cc.ConvTranspose1d(1, 1, 4, 2, 1)
    cb1 = cc.CachedConvTranspose1d(1, 1, 4, 2, 1)

    cb1.weight.data.copy_(b1.weight.data)
    cb1.bias.data.copy_(b1.bias.data)

    b2 = cc.ConvTranspose1d(1, 1, 4, 2, 1)
    cb2 = cc.CachedConvTranspose1d(1, 1, 4, 2, 1)

    cb2.weight.data.copy_(b2.weight.data)
    cb2.bias.data.copy_(b2.bias.data)

    model = cc.AlignBranches(b1, b2, stride=2)
    cmodel = cc.AlignBranches(cb1, cb2, stride=2)

    x = torch.randn(1, 1, 2**14)

    y = sum(model(x))[..., :-cmodel.cumulative_delay]
    cy = sum(cmodel(x))[..., cmodel.cumulative_delay:]

    assert torch.allclose(y, cy, 1e-3, 1e-3)
