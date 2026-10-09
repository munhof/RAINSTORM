"""Session sources and explicit scientific preparation for STORM experiment DAGs.

Catalog registration imports no scientific runtime. Arrays live in worker files;
plans retain only manifests, content hashes and declared recipe parameters.
"""
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from tempfile import mkdtemp
from uuid import uuid4

from storm.experiments import NodeOperation
from storm.adapters import DataAdapter
from storm.experiment_nodes import object_schema
from storm.pipeline import PipelineContext
from storm.pipeline.context import ContentDescriptor


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _source(inputs, config, *, preview=False):
    path = Path(config['manifest_path'])
    if _sha256(path) != config['sha256']:
        raise ValueError('Session manifest changed')
    manifest = json.loads(path.read_text())
    if manifest.get('version') != '1':
        raise ValueError('Unsupported session manifest version')
    sessions = manifest['sessions']
    if len({s['id'] for s in sessions}) != len(sessions):
        raise ValueError('Session identities must be unique across partitions')
    if any(s['partition'] not in ('train', 'validation', 'test') for s in sessions):
        raise ValueError('Every session must declare a partition')
    result = {}
    for partition in ('train', 'validation'):
        selected = [deepcopy(s) for s in sessions if s['partition'] == partition]
        if not selected:
            raise ValueError(f'No sessions declared for {partition}')
        if preview:
            selected = selected[:1]
        for session in selected:
            if set(session) - {'id', 'partition', 'path', 'sha256'}:
                raise ValueError('Unknown session manifest fields')
            source = Path(session['path'])
            if not source.is_absolute():
                source = path.parent / source
            session['path'] = str(source.resolve())
            if _sha256(source) != session['sha256']:
                raise ValueError(f"Source contents changed: {session['id']}")
        if preview:
            for session in selected:
                session['preview_frames'] = 256
        ids = [s['id'] for s in selected]
        result[partition] = PipelineContext(data=selected, dataset_id=config['sha256'],
            metadata={'partition': partition, 'partitions': [partition] * len(ids),
                'sessions': ids, 'observation_ids': ids, 'observation_indices': list(range(len(ids))),
                'task': 'clustering', 'units': 'px', 'manifest_sha256': config['sha256'],
                'time_axis': {'clock': 'pose_frame', 'unit': 'frame',
                              'origin': 'source_frame_id', 'scope': 'session'}},
            schema={'data': ContentDescriptor('session-reference', granularity='session',
                                             identity='metadata.observation_ids', units='px')})
    if preview:
        for context in result.values():
            context.metadata['preview_scope'] = 'sample_only'
    return result


def _source_preview(inputs, config):
    return _source(inputs, config, preview=True)


@lru_cache(maxsize=16)
def _read_array(path, digest):
    import numpy as np
    if digest and _sha256(path) != digest:
        raise ValueError('Prepared artifact contents changed')
    return np.load(path, mmap_mode='r', allow_pickle=False)


@dataclass(frozen=True)
class WindowLineage:
    sessions: tuple[str, ...]
    frame_ids: tuple[tuple[int, ...], ...]
    positions: tuple[tuple[int, int], ...]
    offsets: tuple[int, ...]

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        file_index, center = self.positions[index]
        return [f'{self.sessions[file_index]}:{self.frame_ids[file_index][center + offset]}'
                for offset in self.offsets]

    def __deepcopy__(self, memo):
        return self


@dataclass(frozen=True)
class WindowCollection:
    """Immutable file references. Each returned view is read-only and lazily opened."""
    files: tuple[str, ...]
    positions: tuple[tuple[int, int], ...]
    offsets: tuple[int, ...]
    digests: tuple[str, ...] = ()

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, index):
        import numpy as np
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        file_index, center = self.positions[index]
        values = _read_array(self.files[file_index], self.digests[file_index] if self.digests else '')
        window = values[[center + offset for offset in self.offsets]]
        window.flags.writeable = False
        return window

    def __deepcopy__(self, memo):
        # Immutable references own no open resources; writable arrays are never exposed.
        return self


def _write_array(directory, name, values):
    import numpy as np
    path = directory / f'{name}.npy'
    np.save(path, values, allow_pickle=False)
    return {'path': str(path), 'sha256': _sha256(path)}


def _prepare(inputs, config, artifact_root):
    import numpy as np
    import pandas as pd
    from .reconstruction_v2 import ReconstructPoseV2
    from .preprocessing import RecenterPose, OrientPose
    context = inputs['context']
    calibration = config.get('calibration_ref')
    if calibration:
        if _sha256(calibration['path']) != calibration['sha256']:
            raise ValueError('Calibration contents changed')
        roi = json.loads(Path(calibration['path']).read_text())
        if (config['px_per_cm_x'] != roi['scale'] or config['px_per_cm_y'] != roi['scale']
                or config['flip_y_height'] != roi['frame_shape'][1]):
            raise ValueError('Preparation calibration disagrees with referenced ROI')
    parts, offsets = config['bodyparts'], config['offsets']
    if not parts or len(set(parts)) != len(parts) or any(p not in parts for p in ('body', 'nose')):
        raise ValueError('Declare distinct bodyparts including body and nose')
    if sorted(set(offsets)) != offsets or 0 not in offsets:
        raise ValueError('Window offsets must be sorted, unique and include zero')
    directory = Path(artifact_root) / uuid4().hex
    directory.mkdir(parents=True)
    files, positions, ids, sessions, frames, source_frames_all, audit, refs = [], [], [], [], [], [], [], []
    source_ids = []
    preview_rows, preview_ids = [], []
    feature_names = [f'{part}_{axis}' for part in parts for axis in ('x', 'y')]
    partition = context.metadata['partition']
    for file_index, source in enumerate(context.data):
        if source['partition'] != partition:
            raise ValueError('Preparation cannot cross partitions')
        if _sha256(source['path']) != source['sha256']:
            raise ValueError(f"Source contents changed: {source['id']}")
        read_options = {'start': 0, 'stop': source['preview_frames']} if 'preview_frames' in source else {}
        table = pd.read_hdf(source['path'], **read_options)
        table.columns = ['_'.join(map(str, c[-2:])) if isinstance(c, tuple) else str(c) for c in table.columns]
        values = table[[f'{p}_{axis}' for p in parts for axis in ('x', 'y')]].to_numpy(dtype=np.float32).copy()
        raw_preview = values.copy() if 'preview_frames' in source else None
        likelihoods = table[[p+'_likelihood' for p in parts]].to_numpy(dtype=np.float32)
        frame_ids = table.index.tolist()
        if any(type(f) not in (int, np.int64, np.int32) for f in frame_ids) or len(set(frame_ids)) != len(frame_ids):
            raise ValueError('Pose frames must be distinct integer identities')
        frame_ids = [int(f) for f in frame_ids]
        if frame_ids != sorted(frame_ids):
            raise ValueError('Pose frames must be ordered')
        trusted = np.isfinite(likelihoods) & (likelihoods >= config['threshold'])
        if np.any((values[:, 1::2] >= config['flip_y_height']) & trusted):
            raise ValueError(f"{source['id']}: pose exceeds declared Y frame; verify source resolution")
        values[:, 0::2] /= config['px_per_cm_x']
        values[:, 1::2] = (config['flip_y_height'] - values[:, 1::2]) / config['px_per_cm_y']
        count = len(values)
        pose = PipelineContext(data=values, metadata={'units': 'cm', 'sessions': [source['id']] * count,
            'partitions': [partition] * count, 'segments': ['0'] * count,
            'frames': frame_ids, 'likelihoods': likelihoods})
        recipe = {k: config[k] for k in ('threshold', 'fps', 'max_velocity', 'dist_near',
                                         'dist_far', 'max_outlier_connections', 'med_window', 'sigma')}
        ReconstructPoseV2(**recipe).process(pose)
        reconstruction = pose.state['reconstruction_v2']
        reconstruction_ref = _write_array(directory, f'{file_index}-mask', reconstruction['valid_mask'])
        coordinate_pairs = [[2*i, 2*i+1] for i in range(len(parts))]
        RecenterPose(coordinate_pairs[parts.index('body')], coordinate_pairs, device='cpu').process(pose)
        OrientPose([coordinate_pairs[parts.index('nose')], coordinate_pairs[parts.index('body')]],
                   coordinate_pairs, target_angle_degrees=90, device='cpu').process(pose)
        values = np.asarray(pose.data, dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite prepared coordinates: {source['id']}")
        ref = _write_array(directory, str(file_index), values)
        ref.update(session=source['id'], partition=partition, frames=count,
                   mask=reconstruction_ref, source_sha256=source['sha256'])
        refs.append(ref); files.append(ref['path'])
        source_frames_all.append(tuple(frame_ids)); source_ids.append(source['id'])
        lookup = {frame: i for i, frame in enumerate(frame_ids)}
        window_count = 0
        for i, frame in enumerate(frame_ids):
            source_frames = [frame + offset for offset in offsets]
            if any(f not in lookup for f in source_frames):
                continue
            # WindowCollection indexes by array position; reject discontinuous windows.
            if any(lookup[f] != i + offset for f, offset in zip(source_frames, offsets)):
                continue
            if raw_preview is not None and len(preview_rows) < 5:
                preview_rows.append(raw_preview[i].tolist())
                preview_ids.append(f"{source['id']}:{frame}")
            positions.append((file_index, i)); ids.append(f"{source['id']}:{frame}")
            sessions.append(source['id']); frames.append(frame)
            window_count += 1
        if not window_count:
            raise ValueError(f"No complete windows after preparation: {source['id']}")
        audit.append({'session': source['id'], 'partition': partition, 'frames': count,
            'windows': window_count, 'invalid_after_likelihood': reconstruction['invalid_after_likelihood'],
            'invalid_after_spatial': reconstruction['invalid_after_spatial'], 'mask_ref': reconstruction_ref,
            'calibration': {k: config[k] for k in ('px_per_cm_x', 'px_per_cm_y', 'flip_y_height')},
            'recipe': recipe, 'source_sha256': source['sha256']})
    positions = tuple(positions)
    context.data = WindowCollection(tuple(files), positions, tuple(offsets), tuple(r['sha256'] for r in refs))
    context.targets = None
    if preview_rows:
        context.state['preview_input_context'] = {'rows': preview_rows, 'data': {'length': sum(item['frames'] for item in audit)},
            'metadata': {'units': 'px', 'observation_ids': preview_ids, 'feature_names': feature_names},
            'schema': {'data': {'units': 'px'}}}
    context.metadata.update(units='cm', feature_names=feature_names, observation_ids=ids, sessions=sessions, frames=frames, times=list(frames),
                            observation_indices=list(range(len(ids))), partitions=[partition]*len(ids),
                            prepared_steps=['pose.reconstruct_v2', 'pose.recenter',
                                            'pose.orient_coordinates', 'pose.temporal_windows'])
    context.state['source_validity_mask'] = WindowCollection(
        tuple(r['mask']['path'] for r in refs), positions, (0,),
        tuple(r['mask']['sha256'] for r in refs))
    context.state.update(preparation_audit=audit, window_parents=WindowLineage(
        tuple(source_ids), tuple(source_frames_all), positions, tuple(offsets)))
    context.artifacts['prepared_sessions'] = refs
    context.schema['data'] = ContentDescriptor('float32', ('observation', len(offsets), 2*len(parts)),
        units='cm', granularity='window', identity='metadata.observation_ids', time='metadata.frames')
    return {'context': context}


def _frames(inputs, config):
    """Explicit raw-pixel frame recipe for exploratory supervised inference."""
    import numpy as np
    import pandas as pd
    context = inputs['context']
    partition = context.metadata.get('partition')
    parts = config['bodyparts']
    if not parts or len(set(parts)) != len(parts):
        raise ValueError('Choose distinct bodyparts in an explicit order')
    columns = [f'{part}_{axis}' for part in parts for axis in ('x', 'y')]
    rows, ids, sessions, frames, audits, refs = [], [], [], [], [], []
    for source in context.data:
        if partition not in ('train', 'validation') or source['partition'] != partition:
            raise ValueError('Frame preparation cannot cross a partition or read reserved test')
        if _sha256(source['path']) != source['sha256']:
            raise ValueError('Source contents changed')
        options = {'start': 0, 'stop': source['preview_frames']} if 'preview_frames' in source else {}
        table = pd.read_hdf(source['path'], **options)
        table.columns = ['_'.join(map(str, c[-2:])) if isinstance(c, tuple) else str(c) for c in table.columns]
        frame_ids = table.index.tolist()
        if any(not isinstance(frame, (int, np.integer)) or frame < 0 for frame in frame_ids) or frame_ids != sorted(set(frame_ids)):
            raise ValueError('Frames must have distinct ordered integer identities')
        values = table[columns].to_numpy(dtype=float)
        confidence = table[[part + '_likelihood' for part in parts]].to_numpy(dtype=float)
        valid = np.isfinite(values).all(axis=1) & np.isfinite(confidence).all(axis=1) & (confidence >= config['threshold']).all(axis=1)
        selected = np.flatnonzero(valid)
        if not len(selected):
            raise ValueError(f"No valid frames for {source['id']}")
        rows.extend(values[selected].tolist())
        frames.extend(int(frame_ids[i]) for i in selected)
        ids.extend(f"{source['id']}:{frame_ids[i]}" for i in selected)
        sessions.extend([source['id']] * len(selected))
        audits.append({'session': source['id'], 'frames': len(frame_ids), 'valid': len(selected),
            'discarded': int((~valid).sum()), 'threshold': config['threshold'],
            'discarded_frame_ids': [int(frame_ids[i]) for i in np.flatnonzero(~valid)],
            'recipe': 'raw_pixels_confidence_v1', 'interpolation': False})
        refs.append({'session': source['id'], 'source_sha256': source['sha256']})
    context.data, context.targets = rows, None
    context.metadata.update(partition=partition, partitions=[partition] * len(rows), units='px',
        feature_names=columns, observation_ids=ids, sessions=sessions, segments=list(sessions),
        frames=frames, times=list(frames), observation_indices=list(range(len(rows))),
        prepared_steps=['pose.raw_frames.v1'], exploratory=True)
    context.state['preparation_audit'] = audits
    context.artifacts['prepared_sessions'] = refs
    context.schema['data'] = ContentDescriptor('number', ('observations', len(columns)), units='px',
        identity='metadata.observation_ids', time='metadata.frames')
    return {'context': context}


def _sample(inputs, config):
    import numpy as np
    context = inputs['context']
    count = len(context.data)
    rng = np.random.default_rng(config['seed'])
    selected = sorted(rng.choice(count, size=min(count, config['limit']), replace=False).tolist())
    context.data = [context.data[i] for i in selected]
    for key in ('sessions', 'partitions', 'observation_ids', 'observation_indices', 'frames', 'times'):
        context.metadata[key] = [context.metadata[key][i] for i in selected]
    context.state['window_parents'] = [context.state['window_parents'][i] for i in selected]
    context.state['source_validity_mask'] = [context.state['source_validity_mask'][i] for i in selected]
    context.state['sampling'] = {'seed': config['seed'], 'source_count': count, 'selected': selected}
    return {'context': context}


def register_experiment_nodes(catalog):
    strings = {'type': 'array', 'items': {'type': 'string'}, 'minItems': 1}
    catalog.experiment_nodes.append(NodeOperation('pose.load_frames', '1', {'context': 'context'},
        {'context': 'context'}, object_schema({'bodyparts': strings,
            'threshold': {'type': 'number', 'minimum': 0, 'maximum': 1, 'default': .6}}, ('bodyparts',)),
        _frames, reads=('data', 'metadata'), writes=('data', 'metadata', 'state', 'artifacts', 'schema'),
        descriptor={'label': 'Frames de pose · píxeles',
            'description': 'Receta exploratoria: selecciona coordenadas y descarta frames no finitos o bajo confianza, conservando identidad. Sin interpolación ni equivalencia histórica.'}))
    catalog.data_adapters['pose_sessions'] = DataAdapter('pose_sessions', '1', object_schema({
        'manifest_path': {'type': 'string'}, 'sha256': {'type': 'string'}}, ('manifest_path', 'sha256')),
        lambda config: _source({}, config),
        {'data': {'dtype': 'session-reference', 'granularity': 'session', 'units': 'px',
                  'identity': 'metadata.observation_ids',
                  'time': {'clock': 'pose_frame', 'unit': 'frame', 'origin': 'source_frame_id', 'scope': 'session'}},
         'partitions': ['train', 'validation']},
        outputs=('train', 'validation'), preview=lambda config: _source_preview({}, config))
    properties = {
        'calibration_ref': object_schema({'path': {'type': 'string'}, 'sha256': {'type': 'string'}}, ('path', 'sha256')),
        'bodyparts': strings, 'px_per_cm_x': {'type': 'number', 'minimum': .000001},
        'px_per_cm_y': {'type': 'number', 'minimum': .000001},
        'flip_y_height': {'type': 'number', 'minimum': 1},
        'offsets': {'type': 'array', 'items': {'type': 'integer'}, 'default': list(range(-9, 11))},
        'threshold': {'type': 'number', 'minimum': 0, 'maximum': 1, 'default': .6},
        'fps': {'type': 'number', 'minimum': .000001, 'default': 30.},
        'max_velocity': {'type': 'number', 'minimum': .000001, 'default': 120.},
        'dist_near': {'type': 'number', 'minimum': .000001, 'default': 4.},
        'dist_far': {'type': 'number', 'minimum': .000001, 'default': 12.},
        'max_outlier_connections': {'type': 'integer', 'minimum': 0, 'default': 3},
        'med_window': {'type': 'integer', 'minimum': 1, 'default': 5},
        'sigma': {'type': 'number', 'minimum': .000001, 'default': .6}}
    def prepare(inputs, config):
        root = getattr(catalog, 'experiment_artifact_root', None)
        if root is None:
            root = mkdtemp(prefix='rainstorm-prepared-')
        return _prepare(inputs, config, root)
    def validate_preparation(config):
        from .reconstruction_v2 import ReconstructPoseV2
        parts, offsets = config['bodyparts'], config['offsets']
        if len(set(parts)) != len(parts) or not {'nose', 'body'} <= set(parts):
            raise ValueError('Declare distinct bodyparts including nose and body')
        if sorted(set(offsets)) != offsets or 0 not in offsets:
            raise ValueError('Window offsets must be sorted, unique and include zero')
        ReconstructPoseV2(**{k: config[k] for k in ('threshold', 'fps', 'max_velocity', 'dist_near',
            'dist_far', 'max_outlier_connections', 'med_window', 'sigma')})
    catalog.experiment_nodes.append(NodeOperation('pose.prepare_sessions' , '2', {'context': 'context'},
        {'context': 'context'}, object_schema(properties,
            ('bodyparts', 'px_per_cm_x', 'px_per_cm_y', 'flip_y_height')), prepare,
        reads=('data', 'metadata.partitions'), writes=('data', 'schema', 'metadata', 'state', 'artifacts'), validate_config=validate_preparation))
    catalog.experiment_nodes.append(NodeOperation('pose.sample_windows', '1', {'context': 'context'},
        {'context': 'context'}, object_schema({'limit': {'type': 'integer', 'minimum': 1, 'default': 256},
                                              'seed': {'type': 'integer', 'default': 156}}), _sample,
        reads=('data', 'metadata', 'state.window_parents'), writes=('data', 'metadata', 'state')))
