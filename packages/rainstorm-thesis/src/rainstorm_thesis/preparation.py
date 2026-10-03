"""Resolve declarative pose choices without mutating a recipe or schema."""
from copy import deepcopy


def resolve_preparation_steps(steps, feature_names):
    """Resolve bodypart choices against the features available at each step."""
    current_features = list(feature_names or [])
    resolved = []
    for step in steps:
        item = deepcopy(step)
        step_type = item.get('type')
        config = dict(item.get('config') or {})
        if step_type == 'pose.select_coordinates':
            names = config.get('names')
            if not isinstance(names, list) or not names:
                raise ValueError('Select at least one pose coordinate.')
            missing = [name for name in names if name not in current_features]
            if missing:
                raise ValueError(f'Unknown pose coordinates: {missing}')
            current_features = names
        elif step_type == 'pose.recenter' and 'center_bodypart' in config:
            center = config.get('center_bodypart')
            bodyparts = config.get('bodyparts')
            center_indices = _bodypart_pair(current_features, center)
            pairs = [_bodypart_pair(current_features, name) for name in bodyparts or []]
            config = {'center_indices': center_indices, 'coordinate_pairs': pairs}
            if 'device' in item.get('config', {}):
                config['device'] = item['config']['device']
        elif step_type == 'pose.orient_coordinates' and 'from_bodypart' in config:
            requested_device = config.get('device')
            bodyparts = list(dict.fromkeys(
                name[:-2] for name in current_features
                if name.endswith('_x') and f'{name[:-2]}_y' in current_features))
            reference_pairs = [
                _bodypart_pair(current_features, config.get('from_bodypart')),
                _bodypart_pair(current_features, config.get('toward_bodypart')),
            ]
            coordinate_pairs = [_bodypart_pair(current_features, name)
                                for name in bodyparts]
            config = {
                'reference_pairs': reference_pairs,
                'coordinate_pairs': coordinate_pairs,
                'target_angle_degrees': config.get('target_angle_degrees', 45),
                'degenerate_reference_policy': config.get(
                    'degenerate_reference_policy', 'error'),
            }
            if requested_device is not None:
                config['device'] = requested_device
        elif step_type == 'pose.likelihood_filter' and 'bodyparts' in config:
            pairs = [_bodypart_pair(current_features, name)
                     for name in config.get('bodyparts') or []]
            config = {'threshold': config.get('threshold'), 'coordinate_pairs': pairs}
        if 'config' in item or (isinstance(step_type, str) and step_type.startswith('pose.')):
            item['config'] = config
        resolved.append(item)
    return resolved


def _bodypart_pair(feature_names, bodypart):
    if not isinstance(bodypart, str) or not bodypart:
        raise ValueError('Choose a body part for the pose transformation.')
    names = (f'{bodypart}_x', f'{bodypart}_y')
    if any(name not in feature_names for name in names):
        raise ValueError(f'Body part {bodypart!r} is unavailable at this step.')
    return [feature_names.index(name) for name in names]
