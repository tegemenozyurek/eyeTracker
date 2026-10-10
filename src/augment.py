"""Webcam-style data augmentation for 32x32 eye crops (eyeTrack0.5), on the GPU.

Every time the model sees an eye it gets a randomly degraded version, so it has to learn the eyelid
and not the look of one camera:
  - geometry: mirror, rotation, zoom (MRL crops are framed tighter than our live crops, see Step 9),
    shift (landmark jitter)
  - lighting: brightness, contrast, gamma (dim cars, backlight, infrared vs daylight)
  - optics: Gaussian or motion blur, low resolution (the eye is only a few dozen pixels wide)
  - sensor: Gaussian noise
Same recipe as emotionDetecter's augment_video, scaled to 32 px. The test-time simulated webcam
(src/webcam.py) uses the same kinds of effects with fixed strengths, so eyeTrack0.5 has an advantage
on that test by design: that is the point, but it is stated wherever the numbers are shown.

x: (N, 1, H, W) floats in [0, 1] on any device. Returns the same shape.
"""
import math

import torch
import torch.nn.functional as F


def _blur_kernels(n, k, device, blur_prob):
    """One k x k kernel per image: identity, Gaussian (out of focus) or a line (motion blur)."""
    r = k // 2
    ys, xs = torch.meshgrid(torch.arange(-r, r + 1, device=device, dtype=torch.float32),
                            torch.arange(-r, r + 1, device=device, dtype=torch.float32), indexing="ij")
    sigma = torch.empty(n, 1, 1, device=device).uniform_(0.5, 1.2)
    gauss = torch.exp(-(xs ** 2 + ys ** 2) / (2 * sigma ** 2))
    angle = torch.rand(n, 1, 1, device=device) * math.pi
    length = torch.empty(n, 1, 1, device=device).uniform_(2.0, float(k))
    along = xs * torch.cos(angle) + ys * torch.sin(angle)
    across = -xs * torch.sin(angle) + ys * torch.cos(angle)
    motion = ((across.abs() <= 0.5) & (along.abs() <= length / 2)).float()
    identity = ((xs == 0) & (ys == 0)).float().expand(n, k, k)
    kind = torch.rand(n, 1, 1, device=device)
    kernels = torch.where(kind < blur_prob / 2, gauss, torch.where(kind < blur_prob, motion, identity))
    return (kernels / kernels.sum(dim=(1, 2), keepdim=True)).unsqueeze(1)


def webcam(x, max_rotate=12, max_shift=0.08, scale_range=(0.8, 1.15), blur_prob=0.4, lowres_prob=0.3,
           noise_prob=0.5, max_noise=0.05):
    n, _, h, w = x.shape
    device = x.device

    # geometry: one affine transform per image (scale < 1 zooms out, showing more around the eye)
    flip = torch.where(torch.rand(n, device=device) < 0.5, -1.0, 1.0)
    angle = (torch.rand(n, device=device) * 2 - 1) * math.radians(max_rotate)
    scale = torch.empty(n, device=device).uniform_(*scale_range)
    tx, ty = (torch.rand(2, n, device=device) * 2 - 1) * max_shift * 2
    cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
    theta = torch.stack([torch.stack([cos * flip, -sin, tx], 1), torch.stack([sin * flip, cos, ty], 1)], 1)
    x = F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False), padding_mode="border", align_corners=False)

    # lighting
    mean = x.mean(dim=(2, 3), keepdim=True)
    contrast = torch.empty(n, 1, 1, 1, device=device).uniform_(0.7, 1.3)
    brightness = torch.empty(n, 1, 1, 1, device=device).uniform_(-0.12, 0.12)
    gamma = torch.exp(torch.empty(n, 1, 1, 1, device=device).uniform_(math.log(0.7), math.log(1.6)))
    x = ((x - mean) * contrast + mean + brightness).clamp(1e-4, 1) ** gamma

    # optics: blur, one kernel per image as a grouped convolution
    k = 5
    kernels = _blur_kernels(n, k, device, blur_prob)
    x = F.conv2d(F.pad(x.reshape(1, n, h, w), (k // 2,) * 4, mode="replicate"), kernels, groups=n).reshape(n, 1, h, w)

    # optics: low resolution
    low = torch.rand(n, device=device) < lowres_prob
    if low.any():
        size = int(h * float(torch.empty(1).uniform_(0.35, 0.7)))
        small = F.interpolate(x[low], size=(size, size), mode="bilinear", align_corners=False, antialias=True)
        x[low] = F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False)

    # sensor noise
    sigma = torch.empty(n, 1, 1, 1, device=device).uniform_(0.0, max_noise)
    sigma *= (torch.rand(n, 1, 1, 1, device=device) < noise_prob)
    return (x + torch.randn_like(x) * sigma).clamp(0, 1)


def augment(x, kind):
    if kind == "webcam":
        return webcam(x)
    raise ValueError(f"unknown augmentation {kind!r}")
