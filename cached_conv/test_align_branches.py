import pytest
import cached_conv as cc
import torch
import torch.nn as nn
import copy


def test_residual():
    cc.use_cached_conv(False)
    conv = cc.Conv1d(1, 1, 5, padding=cc.get_padding(5))
    model = cc.AlignBranches(
        conv,
        nn.Identity(),
        delays=[conv.cumulative_delay, 0],
    )

    cc.use_cached_conv(True)
    cconv = cc.Conv1d(1, 1, 5, padding=cc.get_padding(5))
    cmodel = cc.AlignBranches(
        cconv,
        nn.Identity(),
        delays=[cconv.cumulative_delay, 0],
    )

    for p1, p2 in zip(model.parameters(), cmodel.parameters()):
        p2.data.copy_(p1.data)

    x = torch.randn(1, 1, 2**14)

    y = sum(model(x))[..., :-cmodel.cumulative_delay]
    cy = cc.chunk_process(
        lambda x: sum(cmodel(x)),
        x,
        4,
    )[..., cmodel.cumulative_delay:]

    assert torch.allclose(y, cy, 1e-3, 1e-3)


def test_parallel():
    cc.use_cached_conv(False)

    b1 = cc.Conv1d(1, 1, 5, padding=cc.get_padding(5))
    b2 = cc.Conv1d(1, 1, 3, padding=cc.get_padding(3))
    model = cc.AlignBranches(b1, b2)

    cc.use_cached_conv(True)

    cb1 = cc.Conv1d(1, 1, 5, padding=cc.get_padding(5))
    cb2 = cc.Conv1d(1, 1, 3, padding=cc.get_padding(3))
    cmodel = cc.AlignBranches(cb1, cb2)

    for p1, p2 in zip(model.parameters(), cmodel.parameters()):
        p2.data.copy_(p1.data)

    x = torch.randn(1, 1, 2**14)

    y = sum(model(x))[..., :-cmodel.cumulative_delay]
    cy = cc.chunk_process(
        lambda x: sum(cmodel(x)),
        x,
        4,
    )[..., cmodel.cumulative_delay:]

    assert torch.allclose(y, cy, 1e-3, 1e-3)


def test_parallel_stride():
    cc.use_cached_conv(False)

    b1 = cc.Conv1d(1, 1, 5, stride=2, padding=cc.get_padding(5, 2))
    b2 = cc.Conv1d(1, 1, 3, stride=2, padding=cc.get_padding(3, 2))
    model = cc.AlignBranches(b1, b2, stride=.5)

    cc.use_cached_conv(True)

    cb1 = cc.Conv1d(1, 1, 5, stride=2, padding=cc.get_padding(5, 2))
    cb2 = cc.Conv1d(1, 1, 3, stride=2, padding=cc.get_padding(3, 2))
    cmodel = cc.AlignBranches(cb1, cb2, stride=.5)

    for p1, p2 in zip(model.parameters(), cmodel.parameters()):
        p2.data.copy_(p1.data)

    x = torch.randn(1, 1, 2**14)

    y = sum(model(x))[..., :-cmodel.cumulative_delay]
    cy = cc.chunk_process(
        lambda x: sum(cmodel(x)),
        x,
        4,
    )[..., cmodel.cumulative_delay:]

    assert torch.allclose(y, cy, 1e-3, 1e-3)


def test_parallel_transpose():
    cc.use_cached_conv(False)

    b1 = cc.ConvTranspose1d(1, 1, 4, 2, 1)
    b2 = cc.ConvTranspose1d(1, 1, 4, 2, 1)
    model = cc.AlignBranches(b1, b2, stride=2)

    cc.use_cached_conv(True)

    cb1 = cc.ConvTranspose1d(1, 1, 4, 2, 1)
    cb2 = cc.ConvTranspose1d(1, 1, 4, 2, 1)
    cmodel = cc.AlignBranches(cb1, cb2, stride=2)

    for p1, p2 in zip(model.parameters(), cmodel.parameters()):
        p2.data.copy_(p1.data)

    x = torch.randn(1, 1, 2**14)

    y = sum(model(x))[..., :-cmodel.cumulative_delay]
    cy = cc.chunk_process(
        lambda x: sum(cmodel(x)),
        x,
        4,
    )[..., cmodel.cumulative_delay:]

    assert torch.allclose(y, cy, 1e-3, 1e-3)
