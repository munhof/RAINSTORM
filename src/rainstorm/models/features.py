"""Explicit supervised feature recipes, with original-frame window alignment.

Coordinate order, translation and rotation follow RAINSTORM commit 4189f8b,
backend/modeling/{data_handling,aux_functions}.py. Unlike historical edge
padding, this recipe emits only complete windows (recipe version 1).
"""
import numpy as np
from storm.learning import temporal_window_indices


def prepare_supervised_inputs(table, *, bodyparts, frames, sessions, segments,
                              partitions, reserved, offsets, center=None,
                              orientation=None, purpose='inference'):
    if len(bodyparts) != 6 or len(set(bodyparts)) != 6:
        raise ValueError('Specify six distinct bodyparts in model training order')
    if len(table) != len(frames):
        raise ValueError('Frame mapping and pose rows disagree')
    columns = [f'{bp}_{axis}' for bp in bodyparts for axis in ('x','y')]
    values = table[columns].to_numpy(dtype=float).reshape(-1,6,2).copy()
    if center is not None:
        origin = table[[f'{center}_x', f'{center}_y']].to_numpy(dtype=float)
        values -= origin[:,None,:]
    if orientation is not None:
        south, north = orientation
        delta = (table[[f'{north}_x',f'{north}_y']].to_numpy(dtype=float)
                 - table[[f'{south}_x',f'{south}_y']].to_numpy(dtype=float))
        if (np.linalg.norm(delta, axis=1) == 0).any():
            raise ValueError('Orientation references coincide')
        theta = np.pi/4 - np.arctan2(-delta[:,1], -delta[:,0])
        x, y = values[:,:,0].copy(), values[:,:,1].copy()
        values[:,:,0] = x*np.cos(theta[:,None]) - y*np.sin(theta[:,None])
        values[:,:,1] = x*np.sin(theta[:,None]) + y*np.cos(theta[:,None])
    values = values.reshape(-1,12)
    if not np.isfinite(values).all():
        raise ValueError('Feature recipe requires finite coordinates')
    offsets = tuple(offsets)
    windows = temporal_window_indices(frames=frames, sessions=sessions, segments=segments,
        partitions=partitions, reserved=reserved, offsets=offsets, purpose=purpose)
    rows = np.asarray(windows, dtype=int).reshape(-1,len(offsets))
    centers = rows[:, offsets.index(0)].tolist()
    result = values[rows]
    return (result[:,0,:] if offsets == (0,) else result), centers
