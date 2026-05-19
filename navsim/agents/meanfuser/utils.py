import torch
from torch import nn
import torch.nn.functional as F
from enum import IntEnum


HORIZON = 8
ACTION_DIM_DELTA = 4
ACTION_DIM_ORI = 3

x_diff_min = -1.2698211669921875
x_diff_max = 7.475563049316406
x_diff_mean = 2.950225591659546

# Y difference statistics
y_diff_min = -5.012081146240234
y_diff_max = 4.8563690185546875
y_diff_mean = 0.0607292577624321


def diff_traj(traj):
    B, L, _ = traj.shape
    sin = traj[..., -1:].sin()
    cos = traj[..., -1:].cos()
    zero_pad = torch.zeros((B, 1, 1), dtype=traj.dtype, device=traj.device)
    x_diff = traj[..., 0:1].diff(n=1, dim=1, prepend=zero_pad)
    x_diff = x_diff - x_diff_mean
    x_diff_range = max(abs(x_diff_max - x_diff_mean), abs(x_diff_min - x_diff_mean))
    x_diff_norm = x_diff / x_diff_range

    zero_pad = torch.zeros((B, 1, 1), dtype=traj.dtype, device=traj.device)
    y_diff = traj[..., 1:2].diff(n=1, dim=1, prepend=zero_pad)
    y_diff = y_diff - y_diff_mean
    y_diff_range = max(abs(y_diff_max - y_diff_mean), abs(y_diff_min - y_diff_mean))
    y_diff_norm = y_diff / y_diff_range

    return torch.cat([x_diff_norm, y_diff_norm, sin, cos], -1)


def cumsum_traj(norm_trajs):
    sin_values = norm_trajs[..., 2:3]
    cos_values = norm_trajs[..., 3:4]
    heading = torch.atan2(sin_values, cos_values)

    # Denormalize x differences
    x_diff_range = max(abs(x_diff_max - x_diff_mean), abs(x_diff_min - x_diff_mean))
    x_diff = norm_trajs[..., 0:1] * x_diff_range + x_diff_mean

    # Denormalize y differences
    y_diff_range = max(abs(y_diff_max - y_diff_mean), abs(y_diff_min - y_diff_mean))
    y_diff = norm_trajs[..., 1:2] * y_diff_range + y_diff_mean

    # Cumulative sum to get absolute positions
    x = x_diff.cumsum(dim=1)
    y = y_diff.cumsum(dim=1)

    return torch.cat([x, y, heading], -1)


class BoundingBox2DIndex(IntEnum):
    """Intenum for bounding boxes in TransFuser."""

    _X = 0
    _Y = 1
    _HEADING = 2
    _LENGTH = 3
    _WIDTH = 4

    @classmethod
    def size(cls):
        valid_attributes = [
            attribute
            for attribute in dir(cls)
            if attribute.startswith("_") and not attribute.startswith("__") and not callable(getattr(cls, attribute))
        ]
        return len(valid_attributes)

    @classmethod
    @property
    def X(cls):
        return cls._X

    @classmethod
    @property
    def Y(cls):
        return cls._Y

    @classmethod
    @property
    def HEADING(cls):
        return cls._HEADING

    @classmethod
    @property
    def LENGTH(cls):
        return cls._LENGTH

    @classmethod
    @property
    def WIDTH(cls):
        return cls._WIDTH

    @classmethod
    @property
    def POINT(cls):
        # assumes X, Y have subsequent indices
        return slice(cls._X, cls._Y + 1)

    @classmethod
    @property
    def STATE_SE2(cls):
        # assumes X, Y, HEADING have subsequent indices
        return slice(cls._X, cls._HEADING + 1)
