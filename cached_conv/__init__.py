import torch
from .convs import get_padding, CachedSequential, CachedPadding1d
from .convs import Branches, AlignBranches as _AlignBranches
from .convs import Conv1d as _Conv1d, CachedConv1d
from .convs import ConvTranspose1d as _ConvTranspose1d, CachedConvTranspose1d
from .convs import MAX_BATCH_SIZE

USE_BUFFER_CONV = False


def chunk_process(f, x, N):
    x = torch.split(x, x.shape[-1] // N, -1)
    y = torch.cat([f(_x) for _x in x], -1)
    return y


def use_buffer_conv(state: bool):
    global USE_BUFFER_CONV
    USE_BUFFER_CONV = state


def use_cached_conv(state: bool):
    use_buffer_conv(state)


def Conv1d(*args, **kwargs):
    if USE_BUFFER_CONV:
        return CachedConv1d(*args, **kwargs)
    else:
        return _Conv1d(*args, **kwargs)


def ConvTranspose1d(*args, **kwargs):
    if USE_BUFFER_CONV:
        return CachedConvTranspose1d(*args, **kwargs)
    else:
        return _ConvTranspose1d(*args, **kwargs)


def AlignBranches(*args, **kwargs):
    if USE_BUFFER_CONV:
        return _AlignBranches(*args, **kwargs)
    else:
        return Branches(*args, **kwargs)
