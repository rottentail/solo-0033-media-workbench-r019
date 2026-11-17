"""Inverse resampling and edge-weighted composition in scene-linear sRGB."""
import numpy as np


def _linear_table():
    values = np.arange(256, dtype=np.float64) / 255
    return np.where(values <= .04045, values / 12.92,
                    ((values + .055) / 1.055) ** 2.4).astype(np.float32)


def _encode_table():
    # The sRGB transfer curve has a minimum local slope near 0.44, so 16-bit
    # linear quantisation reproduces every 8-bit code exactly.
    values = np.linspace(0., 1., 65536)
    encoded = np.where(values <= .0031308, values * 12.92,
                       1.055 * values ** (1 / 2.4) - .055)
    return np.clip(np.rint(encoded * 255), 0, 255).astype(np.uint8)


LINEAR = _linear_table()
ENCODE = _encode_table()


def _sampling(width, height, mapping):
    x, y, valid, edge = mapping
    x0 = np.floor(x)
    y0 = np.floor(y)
    dx = (x - x0).astype(np.float32)
    dy = (y - y0).astype(np.float32)
    x0 = x0.astype(np.int32)
    y0 = y0.astype(np.int32)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    weight = (valid.astype(np.float32) * edge.astype(np.float32))
    return x0, y0, x1, y1, dx, dy, weight


def prepare(maps, width, height):
    """Freeze per-camera gather indices and feather weights once per delivery."""
    return [_sampling(width, height, mapping) for mapping in maps]


def _pixels(frame, x, y):
    """Gather the three channels at integer coordinates x, y of one picture."""
    return frame[y[..., None], x[..., None], np.arange(3)]


def sample(frame, prepared):
    """Bilinear sample in linear light; return values and edge weight."""
    x0, y0, x1, y1, dx, dy, weight = prepared
    top = (LINEAR[_pixels(frame, x0, y0)] * (1 - dx[..., None])
           + LINEAR[_pixels(frame, x1, y0)] * dx[..., None])
    bottom = (LINEAR[_pixels(frame, x0, y1)] * (1 - dx[..., None])
              + LINEAR[_pixels(frame, x1, y1)] * dx[..., None])
    value = top * (1 - dy[..., None]) + bottom * dy[..., None]
    return value, weight[..., None]


def encode(values):
    index = np.rint(np.clip(values, 0, 1) * 65535).astype(np.int32)
    return ENCODE[index]


def compose(left, right, prepared):
    """Weighted linear-light blend, preserving single-camera pixels exactly."""
    a, wa = sample(left, prepared[0])
    b, wb = sample(right, prepared[1])
    total = wa + wb
    if not np.all(np.isfinite(total)) or not np.all(total > 0):
        raise ValueError('stitch picture contains a pixel outside both cameras')
    merged = (a * wa + b * wb) / total
    # Where only one camera has coverage the result must be that camera's
    # values bit for bit, not a division that could round a code differently.
    only_right = wb[..., 0] == 0
    only_left = wa[..., 0] == 0
    merged[only_right] = a[only_right]
    merged[only_left] = b[only_left]
    return encode(merged)
