"""Immutable two-camera planar calibrations and frame-locked delivery plans."""
import hashlib
import json
import math
import numpy as np
from . import db, media_sources, stitch_geometry


def integer(value, low, high):
    return type(value) is int and low <= value <= high


def _record(conn, table, identity):
    row = conn.execute(f'SELECT * FROM {table} WHERE id=?', (identity,)).fetchone()
    if row is None or hashlib.sha256(row['payload'].encode()).hexdigest() != row['payload_sha256']:
        raise ValueError('stitch record missing or damaged')
    return json.loads(row['payload'])


def save(conn, table, payload):
    # Payloads are pure ASCII (numbers and hashes), so their checksum is stable.
    raw = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False)
    with db.write(conn):
        row = conn.execute(f'INSERT INTO {table}(payload,payload_sha256) VALUES (?,?)',
                           (raw, hashlib.sha256(raw.encode()).hexdigest()))
    result = _record(conn, table, row.lastrowid)
    if result['version'] != 1:
        raise ValueError('unsupported stitch record')
    return dict(result, id=row.lastrowid)


def load(conn, table, identity):
    payload = _record(conn, table, identity)
    if payload['version'] != 1:
        raise ValueError('unsupported stitch record')
    return dict(payload, id=identity)


def source(conn, identity):
    asset = media_sources.get(conn, identity)
    info = json.loads(asset['metadata'])
    videos = [s for s in info['streams'] if s.get('codec_type') == 'video']
    if not videos or not 32 <= videos[0]['width'] <= 640 or not 32 <= videos[0]['height'] <= 360:
        raise ValueError('stitch video must be between 32x32 and 640x360')
    return dict(asset_id=asset['id'], fingerprint=asset['fingerprint'], metadata=asset['metadata'],
                width=videos[0]['width'], height=videos[0]['height'])


def _matrix(value):
    try:
        matrix = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError('stitch record missing or damaged') from exc
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError('stitch record missing or damaged')
    scale = np.linalg.norm(matrix)
    if scale <= 0 or abs(np.linalg.det(matrix)) <= scale ** 3 * 1e-12:
        raise ValueError('stitch record missing or damaged')
    return matrix


def _landmarks(matches, width, height):
    try:
        values = np.asarray(matches, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError('invalid landmark pairs') from exc
    if (values.ndim != 2 or values.shape[1] != 4 or not 6 <= len(values) <= 12
            or not np.isfinite(values).all()):
        raise ValueError('six to twelve finite landmark pairs required')
    for offset in (0, 2):
        if (np.any(values[:, offset:offset + 2] < 0)
                or np.any(values[:, offset] > width - 1)
                or np.any(values[:, offset + 1] > height - 1)):
            raise ValueError('landmarks must be within native pictures')
    return values


def _horizon(matrix, source_width, source_height, target_width, target_height):
    """Reject a registration whose sensor crosses the projection pole.

    A projective map that sends any native pixel centre through infinity has
    no physically supported invertible interpretation on the photographed
    plane, so the calibration cannot be saved. Both the right-to-left map and
    its inverse are checked on the four corners of the sensor they consume.
    """
    def corners(width, height):
        return np.array([[0, 0], [width - 1, 0], [0, height - 1],
                         [width - 1, height - 1]], dtype=float)

    def consistent(matrix, points):
        homogeneous = np.column_stack((points, np.ones(len(points)))) @ matrix.T
        divisor = homogeneous[:, 2]
        if not np.isfinite(divisor).all() or not (np.all(divisor > 1e-8)
                                                  or np.all(divisor < -1e-8)):
            raise ValueError('registration horizon crosses the sensor')
        mapped = homogeneous[:, :2] / divisor[:, None]
        if not np.isfinite(mapped).all():
            raise ValueError('registration horizon crosses the sensor')

    consistent(matrix, corners(source_width, source_height))
    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError('singular planar registration') from exc
    consistent(inverse, corners(target_width, target_height))


def project(matrix, points):
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    if not np.isfinite(homogeneous).all() or np.any(np.abs(homogeneous[:, 2]) < 1e-10):
        raise ValueError('registration maps through infinity')
    return homogeneous[:, :2] / homogeneous[:, 2:]


def profile(conn, left_asset_id, right_asset_id, matches, threshold_px=1.0):
    if (type(threshold_px) not in (int, float) or not math.isfinite(threshold_px)
            or not 0.01 <= threshold_px <= 3):
        raise ValueError('invalid calibration tolerance')
    left, right = source(conn, left_asset_id), source(conn, right_asset_id)
    if (left['width'], left['height']) != (right['width'], right['height']):
        raise ValueError('paired cameras require equal native dimensions')
    values = _landmarks(matches, left['width'], left['height'])
    result = stitch_geometry.calibrate(values, threshold_px)
    _horizon(np.asarray(result['matrix']), right['width'], right['height'],
             left['width'], left['height'])
    # Only verified sources are frozen; a rejected calibration saves nothing.
    media_sources.select(conn, left)
    media_sources.select(conn, right)
    return save(conn, 'stitch_profiles', dict(version=1, left=left, right=right,
                matches=values.tolist(), threshold_px=threshold_px, **result))


def create(conn, profile_id, left_start_ms, right_start_ms, duration_ms, canvas):
    p = load(conn, 'stitch_profiles', profile_id)
    matrix = _matrix(p.get('matrix'))
    frozen = []
    for which in ('left', 'right'):
        clip = p.get(which)
        if (not isinstance(clip, dict) or not isinstance(clip.get('width'), int)
                or not isinstance(clip.get('height'), int)
                or not 32 <= clip['width'] <= 640 or not 32 <= clip['height'] <= 360):
            raise ValueError('stitch record missing or damaged')
        frozen.append(clip)
    if (frozen[0]['width'], frozen[0]['height']) != (frozen[1]['width'], frozen[1]['height']):
        raise ValueError('stitch record missing or damaged')
    if not all(integer(v, 0, 60000) and v % 40 == 0 for v in (left_start_ms, right_start_ms)) or not integer(duration_ms, 40, 30000) or duration_ms % 40:
        raise ValueError('stitch cuts require 25 fps frame boundaries and at most thirty seconds')
    if (not isinstance(canvas, dict) or set(canvas) != {'x', 'y', 'width', 'height'}
            or not all(integer(canvas[k], -1280, 1280) for k in ('x', 'y'))
            or not integer(canvas['width'], 32, 960) or not integer(canvas['height'], 32, 540)):
        raise ValueError('invalid stitch canvas')
    for which, start in (('left', left_start_ms), ('right', right_start_ms)):
        clip = p.get(which)
        if not isinstance(clip, dict) or not isinstance(clip.get('metadata'), str):
            raise ValueError('stitch record missing or damaged')
        try:
            duration = float(json.loads(clip['metadata'])['format']['duration'])
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError('stitch record missing or damaged') from exc
        if start + duration_ms > round(duration * 1000):
            raise ValueError('paired cut exceeds original recording')
    # A saved canvas must be fully covered and the registration physically
    # supported; otherwise no plan record is written at all.
    _horizon(matrix, p['right']['width'], p['right']['height'],
             p['left']['width'], p['left']['height'])
    stitch_geometry.mapping(matrix, canvas, p['left']['width'], p['left']['height'])
    return save(conn, 'stitch_plans', dict(version=1, profile=p, canvas=dict(canvas),
                left_start_ms=left_start_ms, right_start_ms=right_start_ms,
                frames=duration_ms // 40))


def render_profile(conn, identity):
    """Resolve the frozen calibration and revalidate both native sources.

    Returns the frozen profile, its matrix and the resolved left/right clips so
    the caller does not hash identical originals twice before decoding.
    """
    p = load(conn, 'stitch_profiles', identity)
    matrix = _matrix(p.get('matrix'))
    selected = []
    for which in ('left', 'right'):
        clip = p.get(which)
        if not isinstance(clip, dict):
            raise ValueError('stitch record missing or damaged')
        if (not isinstance(clip.get('width'), int) or not isinstance(clip.get('height'), int)
                or not 32 <= clip['width'] <= 640 or not 32 <= clip['height'] <= 360):
            raise ValueError('stitch record missing or damaged')
        selected.append(media_sources.select(conn, clip))
    if p['left']['width'] != p['right']['width'] or p['left']['height'] != p['right']['height']:
        raise ValueError('stitch record missing or damaged')
    _horizon(matrix, p['right']['width'], p['right']['height'],
             p['left']['width'], p['left']['height'])
    return p, matrix, selected


def get(conn, identity):
    return load(conn, 'stitch_plans', identity)
