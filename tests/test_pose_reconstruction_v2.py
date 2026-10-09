import importlib.util
from pathlib import Path
import numpy as np
import pytest
from storm.pipeline import PipelineContext


def context(values, confidence=None):
    n = len(values)
    return PipelineContext(data=values, metadata={'units': 'cm',
        'sessions': ['s'] * n, 'partitions': ['train'] * n, 'segments': ['s:0'] * n,
        'frames': list(range(n)), 'likelihoods': confidence or [[1., 1.]] * n})


def test_reconstruction_matches_recovered_stages_and_keeps_masks():
    from rainstorm_thesis.reconstruction_v2 import ReconstructPoseV2
    values = np.array([[[1. + i/20, 1.], [2.+i/20, 2.]] for i in range(15)])
    ctx = context(values.reshape(15, 4).tolist())
    result = ReconstructPoseV2().process(ctx)
    root = Path(__file__).parents[2] / 'Tesis_Facu/src/rainstorm/domains/pose/preprocessing'
    if not root.exists():
        pytest.skip('Historical checkout is optional for cross-source comparison')
    def load(name):
        spec = importlib.util.spec_from_file_location(name, root / f'{name}.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        return module
    refine, spatial = load('refinement'), load('filtration')
    filtered = spatial.SpatialFiltration(['a', 'b'], mask_until_full_body=False).apply(
        refine.jump_threshold_filter(values, max_v=120, fps=30))
    expected = refine.temporal_smoothing(filtered, med_window=5, sigma=.6)
    np.testing.assert_allclose(np.array(result.data).reshape(15, 2, 2), expected, atol=1e-6)
    assert result.state['reconstruction_v2']['invalid_after_spatial'] == 0
    assert result.state['reconstruction_v2']['valid_mask'] == [[True, True]] * 15


def test_completely_invalid_points_are_diagnosed_instead_of_filled_with_zero():
    from rainstorm_thesis.reconstruction_v2 import ReconstructPoseV2
    ctx = context([[1, 1, 2, 2]] * 10, [[0, 0]] * 10)
    with pytest.raises(ValueError, match='s.*train.*bodypart'):
        ReconstructPoseV2().process(ctx)


def test_reconstruction_never_borrows_valid_points_across_partition_boundary():
    from rainstorm_thesis.reconstruction_v2 import ReconstructPoseV2
    ctx = context([[1, 1, 2, 2]] * 20, [[1, 1]] * 10 + [[0, 0]] * 10)
    ctx.metadata['partitions'] = ['train'] * 10 + ['validation'] * 10
    with pytest.raises(ValueError, match='validation'):
        ReconstructPoseV2().process(ctx)


def test_reconstruction_requires_verified_units():
    from rainstorm_thesis.reconstruction_v2 import ReconstructPoseV2
    ctx = context([[1, 1, 2, 2]] * 10)
    ctx.metadata['units'] = 'px'
    with pytest.raises(ValueError, match='centimeters'):
        ReconstructPoseV2().process(ctx)
