import math
import shutil
from pathlib import Path
from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import IntEnum
from torch.optim.lr_scheduler import _LRScheduler


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        drop=0.0,
        bias=True
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(self.ffn_dropout(hidden))


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


def scaled_dot_product_attention(query, key, value, dropout_p=0.0) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1))
    attn_bias = torch.zeros(query.size(0), 1, L, S, dtype=query.dtype).to(query.device)

    with torch.amp.autocast(device_type="cuda", enabled=False):
        attn_weight = query.float() @ key.float().transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rope=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if rope is not None:
            q = rope(q)
            k = rope(k)

        x = scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)

        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


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

def diff_diff_traj(traj, hist_traj):
    B, L, _ = traj.shape
    sin = traj[..., -1:].sin()
    cos = traj[..., -1:].cos()
    # heading = traj[..., -1:]
    
    diff_xy = traj[..., :2].diff(n=1, dim=1, prepend=hist_traj[:,-2:,:2])
    diff_diff_xy = diff_xy[..., 0:2].diff(n=1, dim=1)
    
    diff_diff_xyh = torch.cat([diff_diff_xy, sin, cos], -1)
    # diff_diff_xyh = torch.tanh(diff_diff_xyh)
    return diff_diff_xyh

def cumsum_cumsum_traj(diff_diff_xyh, hist_traj):
    B, L, _ = diff_diff_xyh.shape
    sin_values = diff_diff_xyh[..., 2:3]
    cos_values = diff_diff_xyh[..., 3:4]
    heading = torch.atan2(sin_values, cos_values)
    # heading = diff_diff_xyh[..., -1:]
    
    initial_diff = hist_traj[:, -1:, :2] - hist_traj[:, -2:-1, :2]
    diff_xy = diff_diff_xyh[..., :2].cumsum(dim=1)
    diff_xy = diff_xy + initial_diff

    initial_xy = hist_traj[:, -1:, :2]
    xy = diff_xy.cumsum(dim=1)
    xy = xy + initial_xy
    return torch.cat([xy, heading], -1)

def copy_py_files_recursive(destination_folder: Union[str, Path]) -> None:
    """
    Recursively copy all .py files from the directory containing this script
    to the destination folder, preserving the directory structure.
    Copies only if the destination file does not exist or the source file's
    modification time is more than 1 second newer than the destination's.

    Args:
        destination_folder: Path to the destination folder (parent directories will be created automatically).
    """
    src_root = Path(__file__).resolve().parent
    dst_root = Path(destination_folder).resolve()
    dst_root.mkdir(parents=True, exist_ok=True)

    # If the destination directory is located inside the source directory,
    # automatically exclude it to avoid processing already copied files.
    exclude_dir = None
    try:
        dst_root.relative_to(src_root)          # Succeeds if dst_root is under src_root
        exclude_dir = dst_root
    except ValueError:
        pass

    copied_count = 0
    for py_file in src_root.rglob("*.py"):
        # Skip files located in the excluded directory
        if exclude_dir and py_file.is_relative_to(exclude_dir):
            continue

        rel_path = py_file.relative_to(src_root)
        dst_path = dst_root / rel_path

        # Create the parent directory for the destination file
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        # Determine whether copying is needed
        need_copy = False
        if not dst_path.exists():
            need_copy = True
        else:
            src_mtime = py_file.stat().st_mtime
            dst_mtime = dst_path.stat().st_mtime
            if src_mtime - dst_mtime > 1:
                need_copy = True

        if need_copy:
            shutil.copy2(py_file, dst_path)     # copy2 preserves original metadata
            print(f"Copied: {rel_path}")
            copied_count += 1

    print(f"Successfully copied {copied_count} .py file(s) to: {destination_folder}")


class WarmupCosLR(_LRScheduler):
    def __init__(
        self, optimizer, min_lr, warmup_epochs, epochs, last_epoch=-1, verbose=False
    ) -> None:
        self.min_lr = min_lr
        self.lr = optimizer.param_groups[0]["lr"]
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        super(WarmupCosLR, self).__init__(optimizer, last_epoch)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {
            key: value for key, value in self.__dict__.items() if key != "optimizer"
        }

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_init_lr(self):
        lr = self.lr / self.warmup_epochs
        return lr

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            lr = self.lr * (self.last_epoch + 1) / self.warmup_epochs
        else:
            lr = self.min_lr + 0.5 * (self.lr - self.min_lr) * (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs)
                    / (self.epochs - self.warmup_epochs)
                )
            )
        if "lr_scale" in self.optimizer.param_groups[0]:
            return [lr * group["lr_scale"] for group in self.optimizer.param_groups]

        return [lr for _ in self.optimizer.param_groups]


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
