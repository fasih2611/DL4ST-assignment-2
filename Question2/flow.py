import cv2
import numpy as np
from torch import Tensor

# Farneback settings fixed by the handout: pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags
FARNEBACK_ARGS = (0.5, 3, 15, 3, 5, 1.2, 0)
MIN_INTENSITY_UINT8 = 51   # intensity > 0.2 on [0, 1]  <=>  > 51 on the uint8 scale
MIN_SPEED = 0.5            # pixels per frame


def flow_directions_and_speeds(videos: Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Pool kept pixels over all consecutive frame pairs of `videos`.

    `videos` is uint8 (N, 20, 1, 64, 64), exactly as stored by MovingMNIST.
    Returns (directions in degrees on [0, 360), speeds in pixels per frame).
    Direction is atan2(-dy, dx): 0 deg = rightward, 90 deg = upward on screen.
    """
    directions, speeds = [], []
    for video in videos.numpy():
        frames = video[:, 0]                        # (20, 64, 64) uint8
        for prev, nxt in zip(frames[:-1], frames[1:]):
            flow = cv2.calcOpticalFlowFarneback(prev, nxt, None, *FARNEBACK_ARGS)
            dx, dy = flow[..., 0], flow[..., 1]     # dx along columns, dy along rows (down)
            speed = np.hypot(dx, dy)
            keep = (prev > MIN_INTENSITY_UINT8) & (nxt > MIN_INTENSITY_UINT8) & (speed > MIN_SPEED)
            angle = np.degrees(np.arctan2(-dy[keep], dx[keep])) % 360.0
            angle[angle >= 360.0] = 0.0             # float rounding can give exactly 360
            directions.append(angle)
            speeds.append(speed[keep])
    return np.concatenate(directions), np.concatenate(speeds)