"""Three blink channels derived from the eyeTrack1 features of one 60 s window (Step 17b).

The dataset's authors detect drowsiness from how people blink (blink duration, eyelid speed). These channels
give the GRU that information explicitly, computed only from features the browser already has:
  closed     1 when the mean eyeBlink score of both eyes is at least 0.5 (and a face was found), else 0
  closed_s   how long the eyes have been closed so far, in seconds, capped at 3 (long blinks, microsleeps)
  speed      how fast the eyelid moved: |change of the mean eyeBlink score| per second
They are computed inside each window, from its first frame, exactly as the browser computes them on its
60 s buffer (web/features.js: blinkChannels, checked by web/blink_channels.test.mjs).
"""
import numpy as np

CHANNELS = ["closed", "closed_s", "speed"]


def blink_channels(x, face, fps):
    """x: (..., T, 13) raw features (columns 0 and 1 = eyeBlink left / right). face: (..., T) bool.
    Returns (..., T, 3) float32."""
    blink = (x[..., 0] + x[..., 1]) / 2
    closed = ((blink >= 0.5) & face).astype(np.float32)
    run = np.zeros(closed.shape, np.float32)
    acc = np.zeros(closed.shape[:-1], np.float32)
    for t in range(closed.shape[-1]):
        acc = (acc + 1 / fps) * closed[..., t]
        run[..., t] = np.minimum(acc, 3.0)
    speed = np.zeros(closed.shape, np.float32)
    both = face[..., 1:] & face[..., :-1]
    speed[..., 1:] = np.abs(blink[..., 1:] - blink[..., :-1]) * fps * both
    return np.stack([closed, run, speed], -1).astype(np.float32)
