"""Bounded decoded spools and lossless planar panorama delivery."""
import os
from pathlib import Path
import tempfile
import numpy as np
from . import stitch_plan, stitch_geometry, stitch_signal, media_sources
from .assets import fingerprint
from .tools import run


def decode(clip, start_ms, frames, destination):
    run('ffmpeg', ['-v', 'error', '-y', '-threads', '1', '-filter_threads', '1',
        '-ss', str(start_ms / 1000), '-i', clip['path'], '-an', '-vf', 'fps=25',
        '-frames:v', str(frames), '-pix_fmt', 'rgb24', '-f', 'rawvideo', str(destination)])
    expected = clip['width'] * clip['height'] * 3 * frames
    if destination.stat().st_size != expected:
        raise ValueError('stitch camera did not provide every planned frame')


def render(conn, identity, destination, at_frame=0, frame_count=None):
    plan = stitch_plan.get(conn, identity)
    frames = plan.get('frames')
    if not stitch_plan.integer(frames, 1, 750) or type(at_frame) is not int:
        raise ValueError('invalid stitch preview range')
    frame_count = frames - at_frame if frame_count is None else frame_count
    if (type(frame_count) is not int
            or not stitch_plan.integer(at_frame, 0, frames - 1)
            or not stitch_plan.integer(frame_count, 1, frames - at_frame)):
        raise ValueError('invalid stitch preview range')
    profile_id = plan['profile'].get('id') if isinstance(plan.get('profile'), dict) else None
    if not stitch_plan.integer(profile_id, 1, 2 ** 62):
        raise ValueError('stitch record missing or damaged')
    p, matrix, selected = stitch_plan.render_profile(conn, profile_id)
    # A plan renders only with the calibration it was accepted with; additive
    # fields such as diagnostics may differ, the physical parameters may not.
    frozen = plan.get('profile')
    if (not isinstance(frozen, dict) or frozen.get('matrix') != p.get('matrix')
            or frozen.get('threshold_px') != p.get('threshold_px')
            or frozen.get('matches') != p.get('matches')):
        raise ValueError('stitch plan does not match its calibration')
    canvas = plan['canvas']
    if (not isinstance(canvas, dict) or set(canvas) != {'x', 'y', 'width', 'height'}
            or not stitch_plan.integer(canvas['width'], 32, 960)
            or not stitch_plan.integer(canvas['height'], 32, 540)
            or not all(stitch_plan.integer(canvas[k], -1280, 1280) for k in ('x', 'y'))):
        raise ValueError('invalid stitch canvas')
    target = Path(destination).expanduser().absolute()
    target = target.parent.resolve() / target.name
    if os.path.lexists(target) or not target.parent.is_dir():
        raise ValueError('stitch destination exists or directory missing')
    maps = stitch_geometry.mapping(matrix, canvas, p['left']['width'], p['left']['height'])
    width, height = p['left']['width'], p['left']['height']
    prepared = stitch_signal.prepare(maps, width, height)
    del maps
    with tempfile.TemporaryDirectory(prefix='.stitch-', dir=target.parent) as directory:
        root = Path(directory)
        paths = [root / 'left.rgb', root / 'right.rgb']
        # Each camera is decoded once, sequentially, to private disk; the
        # whole programme is never held in memory or decoded per preview.
        for which, clip, path in zip(('left', 'right'), selected, paths):
            decode(clip, plan[which + '_start_ms'], plan['frames'], path)
        size = width * height * 3
        raw = root / 'panorama.rgb'
        with paths[0].open('rb') as lh, paths[1].open('rb') as rh, raw.open('wb') as out:
            lh.seek(at_frame * size)
            rh.seek(at_frame * size)
            for _ in range(frame_count):
                pictures = []
                for handle in (lh, rh):
                    data = handle.read(size)
                    if len(data) != size:
                        raise ValueError('truncated stitch camera spool')
                    pictures.append(np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3))
                out.write(stitch_signal.compose(pictures[0], pictures[1], prepared).tobytes())
        # The gather tables are only needed while compositing; release them so
        # the encoder's own buffers have the rest of the container budget.
        del prepared
        result = root / 'panorama.mkv'
        run('ffmpeg', ['-v', 'error', '-y', '-threads', '1', '-filter_threads', '1',
            '-f', 'rawvideo', '-pix_fmt', 'rgb24',
            '-s', f"{canvas['width']}x{canvas['height']}", '-r', '25', '-i', str(raw),
            '-an', '-c:v', 'ffv1', '-level', '3', '-pix_fmt', 'gbrp',
            '-colorspace', 'bt709', '-color_primaries', 'bt709',
            '-color_trc', 'iec61966-2-1', '-color_range', 'pc', str(result)])
        # Revalidate after decoding so a source changed mid-delivery is not
        # published under the accepted plan.
        for clip in selected:
            media_sources.select(conn, clip)
        checksum = fingerprint(result)
        with result.open('rb') as handle:
            os.fsync(handle.fileno())
        os.link(result, target)
    return dict(plan_id=identity, path=str(target), at_frame=at_frame,
                frames=frame_count, sha256=checksum)
