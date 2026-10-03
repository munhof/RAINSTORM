# Ported from Tesis_Facu; copyright Santiago D'hers. See THESIS_LICENSE.
from typing import Any

import numpy as np
from sklearn.decomposition import PCA


def mask_low_confidence(
    coordinates: Any,
    confidences: Any | None,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    coords = np.asarray(coordinates, dtype=np.float64).copy()
    if confidences is None:
        return coords, np.zeros(coords.shape[:2], dtype=bool)

    confidence_values = np.asarray(confidences)
    mask = confidence_values < threshold
    coords[mask] = np.nan
    return coords, mask


def interpolate_masked(coordinates: Any, mask: Any) -> np.ndarray:
    coords = np.asarray(coordinates, dtype=np.float64).copy()
    mask_values = np.asarray(mask, dtype=bool)
    frame_index = np.arange(coords.shape[0])

    for keypoint in range(coords.shape[1]):
        masked_frames = mask_values[:, keypoint] | np.isnan(coords[:, keypoint]).any(
            axis=1
        )
        valid_frames = ~masked_frames
        for dim in range(coords.shape[2]):
            values = coords[:, keypoint, dim]
            valid_dim = valid_frames & ~np.isnan(values)
            if valid_dim.all():
                continue
            if not valid_dim.any():
                coords[:, keypoint, dim] = 0.0
                continue
            coords[:, keypoint, dim] = np.interp(
                frame_index,
                frame_index[valid_dim],
                values[valid_dim],
            )

    return coords


def add_initialization_noise(
    coordinates: Any,
    scale: float = 0.01,
    seed: int = 156,
) -> np.ndarray:
    coords = np.asarray(coordinates, dtype=np.float64)
    rng = np.random.default_rng(seed)
    return coords + rng.normal(loc=0.0, scale=scale, size=coords.shape)


def flatten_keypoints(coordinates: Any) -> np.ndarray:
    coords = np.asarray(coordinates, dtype=np.float64)
    return coords.reshape(coords.shape[0], -1)


def fit_pose_pca(
    coordinates: Any,
    n_components: int | float | None,
    whiten: bool = True,
    seed: int = 156,
) -> tuple[PCA, np.ndarray]:
    flattened = flatten_keypoints(coordinates)
    pca = PCA(n_components=n_components, whiten=whiten, random_state=seed)
    projected = pca.fit_transform(flattened)
    return pca, projected


def apply_pose_pca(coordinates: Any, pca: PCA) -> np.ndarray:
    return pca.transform(flatten_keypoints(coordinates))
