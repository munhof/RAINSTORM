"""RAINSTORM backend public entry points."""


def __getattr__(name):
    # STORM is an optional Python 3.11+ runtime, separate from legacy Python 3.9.
    if name == 'build_pose_pipeline':
        from .pose_pipeline import build_pose_pipeline
        return build_pose_pipeline
    raise AttributeError(name)
