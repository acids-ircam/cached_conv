import torch
import torch.nn as nn


def get_padding(kernel_size, stride=1, dilation=1, mode="centered"):
    """
    Computes 'same' padding given a kernel size, stride an dilation.

    Parameters
    ----------

    kernel_size: int
        kernel_size of the convolution

    stride: int
        stride of the convolution

    dilation: int
        dilation of the convolution

    mode: str
        either "centered", "causal" or "anticausal"
    """
    if kernel_size == 1: return (0, 0)
    p = (kernel_size - 1) * dilation + 1
    half_p = p // 2
    if mode == "centered":
        p_right = half_p
        p_left = half_p
    elif mode == "causal":
        p_right = 0
        p_left = 2 * half_p
    elif mode == "anticausal":
        p_right = 2 * half_p
        p_left = 0
    else:
        raise Exception(f"Padding mode {mode} is not valid")
    return (p_left, p_right)


class CachedSequential(nn.Sequential):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.future_compensation = 0


class CachedPadding1d(nn.Module):
    def __init__(self, padding, crop=False):
        raise Exception(
            "Realtime export is not ready yet. Article coming soon !")


class CachedConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        raise Exception(
            "Realtime export is not ready yet. Article coming soon !")


class CachedConvTranspose1d(nn.ConvTranspose1d):
    def __init__(self, *args, **kwargs):
        raise Exception(
            "Realtime export is not ready yet. Article coming soon !")


class Conv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        self._pad = kwargs.get("padding", (0, 0))
        kwargs["padding"] = 0
        super().__init__(*args, **kwargs)
        self.future_compensation = 0

    def forward(self, x):
        x = nn.functional.pad(x, self._pad)
        return nn.functional.conv1d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class AlignBranches(nn.Module):
    def __init__(self, *branches, futures=None):
        super().__init__()
        self.branches = nn.ModuleList(branches)

    def forward(self, x):
        outs = []
        for branch in self.branches:
            outs.append(branch(x))
        return outs
