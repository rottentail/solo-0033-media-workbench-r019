"""Robust planar registration in native pixel-centre coordinates."""
import numpy as np


def project(matrix, points):
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    if np.any(np.abs(homogeneous[:, 2]) < 1e-10):
        raise ValueError('registration maps through infinity')
    return homogeneous[:, :2] / homogeneous[:, 2:]


def fit(source, target):
    coordinates = np.column_stack((source, np.ones(len(source))))
    coefficients, _, rank, _ = np.linalg.lstsq(coordinates, target, rcond=None)
    if rank < 3:
        raise ValueError('landmarks do not determine a planar registration')
    matrix = np.vstack((coefficients.T, [0., 0., 1.]))
    if abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError('singular planar registration')
    return matrix


def errors(matrix, source, target):
    forward = np.linalg.norm(project(matrix,source)-target, axis=1)
    backward = np.linalg.norm(project(np.linalg.inv(matrix),target)-source, axis=1)
    return np.maximum(forward, backward)


def calibrate(matches, threshold):
    values = np.asarray(matches, dtype=float)
    if values.ndim != 2 or values.shape[1] != 4 or not 6 <= len(values) <= 12 or not np.isfinite(values).all():
        raise ValueError('six to twelve finite [right_x,right_y,left_x,left_y] matches required')
    source, target = values[:,:2], values[:,2:]
    matrix = fit(source, target)
    distances = errors(matrix, source, target)
    return dict(matrix=matrix.tolist(), inliers=list(range(len(values))),
                max_error=float(distances.max()))


def mapping(matrix, canvas, width, height):
    x, y, w, h = (canvas[k] for k in ('x','y','width','height'))
    yy, xx = np.mgrid[y:y+h,x:x+w]
    world = np.column_stack((xx.ravel(),yy.ravel()))
    right = project(np.linalg.inv(np.asarray(matrix)),world)
    maps=[]
    for points in (world,right):
        px, py = points[:,0].reshape(h,w),points[:,1].reshape(h,w)
        valid=(px>=-1e-7)&(py>=-1e-7)&(px<=width-1+1e-7)&(py<=height-1+1e-7)
        px,py=np.clip(px,0,width-1),np.clip(py,0,height-1)
        edge=np.maximum(1e-6,np.minimum.reduce((px+1,py+1,width-px,height-py)))
        maps.append((px,py,valid,edge))
    if not np.all(maps[0][2]|maps[1][2]):
        raise ValueError('canvas contains pixels outside both cameras')
    return maps
