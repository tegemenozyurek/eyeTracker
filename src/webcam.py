"""Simulated webcam: deterministic degradations of eye crops, for testing (same recipe as
emotionDetecter's benchmark, scaled to 32x32 eyes).

Real webcams in a car add dim light, motion blur, low resolution (the eye is only a few dozen
pixels wide), sensor noise and a jittery landmark detector. Each condition can be tested alone
or all combined; a fixed random generator gives every model exactly the same degraded images.

x: (N, 1, H, W) float tensor in [0, 1] on the CPU.
"""
import math

import torch
import torch.nn.functional as F


def dim_light(x, g):
    return ((x - 0.5) * 0.6 + 0.4).clamp(1e-4, 1) ** 1.8


def motion_blur(x, g):
    n, _, h, w = x.shape
    k = max(3, round(7 * h / 64) | 1)  # same blur relative to the image size as emotionDetecter's 7 px at 64 px
    r = torch.arange(k) - k // 2
    ys, xs = torch.meshgrid(r.float(), r.float(), indexing="ij")
    angle = torch.rand(n, 1, 1, generator=g) * math.pi
    across = -xs * torch.sin(angle) + ys * torch.cos(angle)
    kern = (across.abs() <= 0.5).float()
    kern = (kern / kern.sum(dim=(1, 2), keepdim=True)).unsqueeze(1)
    padded = F.pad(x.reshape(1, n, h, w), (k // 2,) * 4, mode="replicate")
    return F.conv2d(padded, kern, groups=n).reshape(n, 1, h, w)


def low_resolution(x, g):
    h = x.shape[-1]
    small = F.interpolate(x, size=(round(h * 0.4),) * 2, mode="bilinear", align_corners=False, antialias=True)
    return F.interpolate(small, size=(h, h), mode="bilinear", align_corners=False)


def sensor_noise(x, g):
    return (x + torch.randn(x.shape, generator=g) * 0.05).clamp(0, 1)


def detector_jitter(x, g):
    """The eye crop lands slightly rotated, shifted and scaled, as when landmarks wobble."""
    n = x.shape[0]
    angle = (torch.rand(n, generator=g) * 2 - 1) * math.radians(8)
    scale = 1 + (torch.rand(n, generator=g) * 2 - 1) * 0.08
    tx, ty = (torch.rand(2, n, generator=g) * 2 - 1) * 0.05 * 2
    cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
    return F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False),
                         padding_mode="border", align_corners=False)


CONDITIONS = {
    "clean": [],
    "dim light": [dim_light],
    "motion blur": [motion_blur],
    "low resolution": [low_resolution],
    "sensor noise": [sensor_noise],
    "detector jitter": [detector_jitter],
    "all combined": [detector_jitter, dim_light, motion_blur, low_resolution, sensor_noise],
}


def degrade(x, condition, seed=0):
    g = torch.Generator().manual_seed(seed)
    for step in CONDITIONS[condition]:
        x = step(x, g)
    return x
