"""Robust planar registration in native pixel-centre coordinates."""
import itertools
import numpy as np


def _normalised(points):
    """Hartley normalisation keeps the projective fit well conditioned."""
    centre = points.mean(axis=0)
    shifted = points - centre
    scale = float(np.mean(np.linalg.norm(shifted, axis=1)))
    if not np.isfinite(scale) or scale <= 1e-9:
        raise ValueError('landmarks do not determine a planar registration')
    factor = np.sqrt(2.0) / scale
    transform = np.array([[factor, 0.0, -factor * centre[0]],
                          [0.0, factor, -factor * centre[1]],
                          [0.0, 0.0, 1.0]])
    return shifted * factor, transform


def _spread(points):
    """Reject coincident and collinear landmarks, which cannot fix a plane."""
    centred = points - points.mean(axis=0)
    singular = np.linalg.svd(centred, compute_uv=False)
    if (not np.all(np.isfinite(singular)) or singular[0] <= 1e-9
            or singular[-1] <= singular[0] * 1e-3):
        raise ValueError('landmarks do not span a plane')


def project(matrix, points):
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    if not np.isfinite(homogeneous).all() or np.any(np.abs(homogeneous[:, 2]) < 1e-10):
        raise ValueError('registration maps through infinity')
    return homogeneous[:, :2] / homogeneous[:, 2:]


def fit(source, target):
    """Least-squares homography from native source to native target points."""
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if (source.ndim != 2 or target.ndim != 2 or source.shape != target.shape
            or source.shape[1] != 2 or len(source) < 4):
        raise ValueError('four or more paired landmark coordinates required')
    if not (np.isfinite(source).all() and np.isfinite(target).all()):
        raise ValueError('landmarks must be finite')
    _spread(source)
    _spread(target)
    normal_source, source_transform = _normalised(source)
    normal_target, target_transform = _normalised(target)
    x, y = normal_source[:, 0], normal_source[:, 1]
    u, v = normal_target[:, 0], normal_target[:, 1]
    n = len(source)
    design = np.zeros((2 * n, 9))
    design[0::2, 0:3] = np.column_stack((x, y, np.ones(n)))
    design[1::2, 3:6] = np.column_stack((x, y, np.ones(n)))
    design[0::2, 6:9] = np.column_stack((-u * x, -u * y, -u))
    design[1::2, 6:9] = np.column_stack((-v * x, -v * y, -v))
    _, spectrum, right = np.linalg.svd(design, full_matrices=True)
    # Eight independent equations leave exactly one projective null direction.
    # With four pairs the design itself has eight rows, so the null vector only
    # appears in the full nine-vector basis. Fewer independent equations leave
    # a family of registrations and therefore no determined plane.
    if not np.all(np.isfinite(spectrum)) or spectrum[0] <= 0:
        raise ValueError('landmarks do not determine a planar registration')
    rank = int(np.count_nonzero(spectrum > spectrum[0] * 1e-9))
    if rank < 8:
        raise ValueError('landmarks do not determine a planar registration')
    matrix = np.linalg.inv(target_transform) @ right[-1].reshape(3, 3) @ source_transform
    if not np.isfinite(matrix).all():
        raise ValueError('landmarks do not determine a planar registration')
    scale = np.linalg.norm(matrix)
    if scale <= 0 or abs(np.linalg.det(matrix)) <= scale ** 3 * 1e-12:
        raise ValueError('singular planar registration')
    # A consistent plane keeps every fitted landmark finite and inside the
    # numerically meaningful range; a sign flip or pole is not a registration.
    fitted = project(matrix, source)
    if not np.isfinite(fitted).all() or np.max(np.abs(fitted)) > 1e9:
        raise ValueError('landmarks do not determine a planar registration')
    return matrix


def errors(matrix, source, target):
    """Per-pair bidirectional pixel error in native coordinates."""
    matrix = np.asarray(matrix, dtype=float)
    inverse = np.linalg.inv(matrix)
    forward = np.linalg.norm(project(matrix, source) - target, axis=1)
    backward = np.linalg.norm(project(inverse, target) - source, axis=1)
    return np.maximum(forward, backward)


def calibrate(matches, threshold):
    """Pick the largest landmark set that agrees on one plane, then refit.

    Enumerating subsets of six to twelve pairs is bounded and deterministic,
    and scanning from the largest size down stops at the first consistent
    majority. Requiring every accepted pair to fit within the bidirectional
    pixel tolerance means a few mislabels cannot drag the calibration.
    """
    try:
        values = np.asarray(matches, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError('six to twelve finite [right_x,right_y,left_x,left_y] matches required') from exc
    if (values.ndim != 2 or values.shape[1] != 4 or not 6 <= len(values) <= 12
            or not np.isfinite(values).all()):
        raise ValueError('six to twelve finite [right_x,right_y,left_x,left_y] matches required')
    if (type(threshold) not in (int, float) or not np.isfinite(threshold)
            or threshold <= 0):
        raise ValueError('invalid calibration tolerance')
    source, target = values[:, :2], values[:, 2:]
    count = len(values)
    chosen = None
    matrix = None
    # Descending size order stops at the largest consensus, which is also the
    # only one that can satisfy the majority requirement.
    for size in range(count, 5, -1):
        candidates = {}
        for subset in itertools.combinations(range(count), size):
            index = list(subset)
            try:
                candidate = fit(source[index], target[index])
                distances = errors(candidate, source[index], target[index])
            except (ValueError, np.linalg.LinAlgError):
                continue
            if np.all(distances <= threshold):
                candidates[subset] = (float(distances.max()), candidate)
        if candidates:
            chosen = min(candidates, key=lambda c: (candidates[c][0], c))
            matrix = candidates[chosen][1]
            break
    if chosen is None:
        raise ValueError('insufficient reliable landmark consensus')
    if 2 * len(chosen) <= count:
        raise ValueError('insufficient reliable landmark consensus')
    inliers = list(chosen)
    distances = errors(matrix, source, target)
    if (not np.all(np.isfinite(distances)) or np.any(distances[inliers] > threshold)
            or len(inliers) < 6 or 2 * len(inliers) <= count):
        raise ValueError('insufficient reliable landmark consensus')
    # The accepted correspondences themselves must determine the plane.
    _spread(source[inliers])
    _spread(target[inliers])
    return dict(matrix=matrix.tolist(), inliers=inliers,
                max_error=float(max(distances[i] for i in inliers)),
                errors=[float(value) for value in distances])


def mapping(matrix, canvas, width, height):
    """Inverse maps and edge weights for every canvas pixel centre.

    Arrays are returned in single precision: a float32 canvas coordinate
    resolves well below a thousandth of a native pixel, and compact maps keep a
    maximum-size canvas inside the delivery machine's memory budget.
    """
    x, y, w, h = (canvas[k] for k in ('x', 'y', 'width', 'height'))
    if not all(isinstance(value, (int, float)) and np.isfinite(value)
               for value in (x, y, w, h)) or w <= 0 or h <= 0:
        raise ValueError('invalid stitch canvas')
    try:
        inverse = np.linalg.inv(np.asarray(matrix, dtype=float))
    except np.linalg.LinAlgError as exc:
        raise ValueError('singular planar registration') from exc
    yy, xx = np.mgrid[y:y + h, x:x + w]
    world = np.column_stack((xx.ravel(), yy.ravel())) * 1.0
    maps = []
    for points in (world, project(inverse, world)):
        px = points[:, 0].reshape(h, w)
        py = points[:, 1].reshape(h, w)
        valid = ((px >= -1e-7) & (py >= -1e-7)
                 & (px <= width - 1 + 1e-7) & (py <= height - 1 + 1e-7))
        # Uncovered pixels get zero weight; they never contribute and never
        # fall back to a duplicated edge sample.
        px = np.where(valid, np.clip(px, 0, width - 1), 0.0).astype(np.float32)
        py = np.where(valid, np.clip(py, 0, height - 1), 0.0).astype(np.float32)
        edge = np.maximum(1e-6, np.minimum.reduce(
            (px + 1, py + 1, width - px, height - py))).astype(np.float32)
        maps.append((px, py, valid, edge))
        del points
    del world
    if not np.all(maps[0][2] | maps[1][2]):
        raise ValueError('canvas contains pixels outside both cameras')
    return maps
