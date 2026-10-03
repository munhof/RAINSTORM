# Ported from Tesis_Facu; copyright Santiago D'hers. See THESIS_LICENSE.
from dataclasses import dataclass

import numpy as np


@dataclass
class KPMSInitResult:
    """Output of KPMS temporal initialization."""

    state_sequence: np.ndarray
    transition_matrix: np.ndarray
    ar_coefficients: np.ndarray | None
    ar_intercepts: np.ndarray | None
    log_likelihood: float | None
    ar_residual_covariances: np.ndarray | None = None
    is_simplified: bool = True
    backend: str = "gaussian_hmm"


@dataclass
class KeypointMoseqInitResult:
    pca_explained_variance: np.ndarray
    actual_latent_dim: int
    init_result: KPMSInitResult
    mask_stats: dict


@dataclass
class KeypointMoseqFitResult:
    init_result: KeypointMoseqInitResult
    syllable_labels: np.ndarray
    transition_matrix: np.ndarray
    state_usage: np.ndarray
    state_durations: dict


@dataclass
class KeypointMoseqOutput:
    syllable_labels: np.ndarray
    latent_pose: np.ndarray
    state_probabilities: np.ndarray | None
