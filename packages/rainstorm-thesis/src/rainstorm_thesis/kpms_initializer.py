# Ported from Tesis_Facu; copyright Santiago D'hers. See THESIS_LICENSE.
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from tqdm.auto import tqdm

from .kpms_config import KeypointMoseqConfig
from .kpms_outputs import KPMSInitResult


@dataclass
class _ARHMMBackend:
    coefficients: np.ndarray
    intercepts: np.ndarray
    covariances: np.ndarray
    start_logprob: np.ndarray
    transition_logprob: np.ndarray
    ar_order: int


class KPMSInitializer:
    """Minimal AR-HMM initializer over latent pose trajectories."""

    def __init__(self, config: KeypointMoseqConfig):
        self.config = config
        self.model_: _ARHMMBackend | None = None

    def fit(
        self,
        pca_trajectory: Any,
        sequence_lengths: list[int] | None = None,
    ) -> KPMSInitResult:
        trajectory = self._validate_trajectory(pca_trajectory)
        lengths = self._validate_sequence_lengths(trajectory, sequence_lengths)
        design, target, valid_mask = self._build_ar_design_matrix(trajectory, lengths)
        labels = self._initialize_labels(target)

        backend: _ARHMMBackend | None = None
        transition_matrix = self._initial_transition_matrix()
        log_likelihood = None

        iterator = range(self.config.num_iters)
        progress = None
        if self.config.show_progress:
            progress = tqdm(
                iterator,
                total=self.config.num_iters,
                desc="KPMS AR-HMM iterations",
                unit="iter",
                leave=False,
            )
            iterator = progress

        try:
            for _ in iterator:
                coefficients, intercepts, covariances = self._estimate_ar_parameters(
                    design,
                    target,
                    labels,
                )
                transition_matrix = self._estimate_transition_matrix(labels, self._valid_lengths(lengths))
                start_logprob = self._estimate_start_logprob(labels, self._valid_lengths(lengths))
                transition_logprob = np.log(np.maximum(transition_matrix, 1e-12))
                start_logprob = np.log(np.maximum(start_logprob, 1e-12))
                backend = _ARHMMBackend(
                    coefficients=coefficients,
                    intercepts=intercepts,
                    covariances=covariances,
                    start_logprob=start_logprob,
                    transition_logprob=transition_logprob,
                    ar_order=self.config.ar_order,
                )
                emissions = self._log_ar_emissions(design, target, backend)
                decoded = self._decode_sequences(emissions, self._valid_lengths(lengths), backend)
                log_likelihood = self._sequence_log_likelihood(
                    emissions,
                    self._valid_lengths(lengths),
                    backend,
                )
                if progress is not None:
                    progress.set_postfix(log_likelihood=f"{log_likelihood:.3g}")
                if np.array_equal(decoded, labels):
                    labels = decoded
                    break
                labels = decoded
        finally:
            if progress is not None:
                progress.close()

        assert backend is not None
        self.model_ = backend
        full_labels = self._expand_labels(labels, valid_mask, lengths)
        return KPMSInitResult(
            state_sequence=full_labels,
            transition_matrix=transition_matrix,
            ar_coefficients=backend.coefficients,
            ar_intercepts=backend.intercepts,
            log_likelihood=log_likelihood,
            ar_residual_covariances=backend.covariances,
            is_simplified=False,
            backend="ar_hmm",
        )

    def predict(
        self,
        pca_trajectory: Any,
        sequence_lengths: list[int] | None = None,
    ) -> np.ndarray:
        if self.model_ is None:
            raise ValueError("KPMSInitializer must be fitted before predict.")
        trajectory = self._validate_trajectory(pca_trajectory)
        lengths = self._validate_sequence_lengths(trajectory, sequence_lengths)
        design, target, valid_mask = self._build_ar_design_matrix(trajectory, lengths)
        emissions = self._log_ar_emissions(design, target, self.model_)
        labels = self._decode_sequences(emissions, self._valid_lengths(lengths), self.model_)
        return self._expand_labels(labels, valid_mask, lengths)

    def predict_proba(
        self,
        pca_trajectory: Any,
        sequence_lengths: list[int] | None = None,
    ) -> np.ndarray:
        if self.model_ is None:
            raise ValueError("KPMSInitializer must be fitted before predict.")
        trajectory = self._validate_trajectory(pca_trajectory)
        lengths = self._validate_sequence_lengths(trajectory, sequence_lengths)
        design, target, valid_mask = self._build_ar_design_matrix(trajectory, lengths)
        emissions = self._log_ar_emissions(design, target, self.model_)
        posterior = self._posterior_sequences(emissions, self._valid_lengths(lengths), self.model_)
        return self._expand_probabilities(posterior, valid_mask, lengths)

    def _initialize_labels(self, target: np.ndarray) -> np.ndarray:
        return KMeans(
            n_clusters=self.config.num_states,
            n_init="auto",
            random_state=self.config.seed,
        ).fit_predict(target)

    def _estimate_ar_parameters(
        self,
        design: np.ndarray,
        target: np.ndarray,
        labels: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_states = self.config.num_states
        latent_dim = target.shape[1]
        predictor_dim = design.shape[1]
        ridge = 1e-6

        global_coefficients, global_intercepts, global_covariance = self._fit_regression(
            design,
            target,
            ridge,
        )

        coefficients = np.empty((n_states, latent_dim, predictor_dim), dtype=np.float64)
        intercepts = np.empty((n_states, latent_dim), dtype=np.float64)
        covariances = np.empty((n_states, latent_dim, latent_dim), dtype=np.float64)

        min_samples = max(self.config.ar_order + 1, predictor_dim + 1)
        for state in range(n_states):
            mask = labels == state
            if mask.sum() < min_samples:
                coefficients[state] = global_coefficients
                intercepts[state] = global_intercepts
                covariances[state] = global_covariance
                continue
            state_coefficients, state_intercepts, state_covariance = self._fit_regression(
                design[mask],
                target[mask],
                ridge,
            )
            coefficients[state] = state_coefficients
            intercepts[state] = state_intercepts
            covariances[state] = state_covariance
        return coefficients, intercepts, covariances

    @staticmethod
    def _fit_regression(
        design: np.ndarray,
        target: np.ndarray,
        ridge: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        predictor_dim = design.shape[1]
        augmented = np.column_stack([design, np.ones(design.shape[0], dtype=np.float64)])
        reg = np.eye(predictor_dim + 1, dtype=np.float64) * ridge
        reg[-1, -1] = 0.0
        beta = np.linalg.solve(augmented.T @ augmented + reg, augmented.T @ target)
        prediction = augmented @ beta
        residuals = target - prediction
        covariance = residuals.T @ residuals / max(residuals.shape[0], 1)
        covariance += np.eye(target.shape[1], dtype=np.float64) * 1e-6
        coefficients = beta[:-1].T
        intercepts = beta[-1]
        return coefficients, intercepts, covariance

    def _initial_transition_matrix(self) -> np.ndarray:
        n_states = self.config.num_states
        transition_counts = np.ones((n_states, n_states), dtype=np.float64)
        transition_counts[np.diag_indices(n_states)] += self.config.kappa
        return transition_counts / transition_counts.sum(axis=1, keepdims=True)

    def _estimate_transition_matrix(
        self,
        labels: np.ndarray,
        sequence_lengths: list[int],
    ) -> np.ndarray:
        n_states = self.config.num_states
        counts = np.ones((n_states, n_states), dtype=np.float64)
        if self.config.use_sticky_prior:
            beta_hat = self._estimate_global_occupancy(labels)
            for state in range(n_states):
                counts[state] += self.config.sticky_alpha * beta_hat
        counts[np.diag_indices(n_states)] += self.config.kappa
        start = 0
        for length in sequence_lengths:
            stop = start + length
            sequence = labels[start:stop]
            for previous, current in zip(sequence[:-1], sequence[1:]):
                counts[int(previous), int(current)] += 1.0
            start = stop
        return counts / counts.sum(axis=1, keepdims=True)

    def _estimate_global_occupancy(self, labels: np.ndarray) -> np.ndarray:
        counts = np.bincount(labels, minlength=self.config.num_states).astype(np.float64)
        counts += 1e-6
        return counts / counts.sum()

    def _estimate_start_logprob(
        self,
        labels: np.ndarray,
        sequence_lengths: list[int],
    ) -> np.ndarray:
        counts = np.ones(self.config.num_states, dtype=np.float64)
        start = 0
        for length in sequence_lengths:
            if length <= 0:
                continue
            counts[int(labels[start])] += 1.0
            start += length
        return counts / counts.sum()

    def _log_ar_emissions(
        self,
        design: np.ndarray,
        target: np.ndarray,
        backend: _ARHMMBackend,
    ) -> np.ndarray:
        n_samples = target.shape[0]
        n_states = backend.coefficients.shape[0]
        emissions = np.empty((n_samples, n_states), dtype=np.float64)
        for state in range(n_states):
            mean = design @ backend.coefficients[state].T + backend.intercepts[state]
            emissions[:, state] = self._log_multivariate_normal(
                target,
                mean,
                backend.covariances[state],
            )
        return emissions

    @staticmethod
    def _log_multivariate_normal(
        target: np.ndarray,
        mean: np.ndarray,
        covariance: np.ndarray,
    ) -> np.ndarray:
        diff = target - mean
        sign, logdet = np.linalg.slogdet(covariance)
        if sign <= 0:
            covariance = covariance + np.eye(covariance.shape[0], dtype=np.float64) * 1e-6
            _, logdet = np.linalg.slogdet(covariance)
        precision = np.linalg.inv(covariance)
        quadratic = np.einsum("ni,ij,nj->n", diff, precision, diff)
        dim = target.shape[1]
        return -0.5 * (quadratic + logdet + dim * np.log(2.0 * np.pi))

    def _decode_sequences(
        self,
        emissions: np.ndarray,
        sequence_lengths: list[int],
        backend: _ARHMMBackend,
    ) -> np.ndarray:
        labels = np.empty(emissions.shape[0], dtype=np.int64)
        start = 0
        for length in sequence_lengths:
            stop = start + length
            labels[start:stop] = self._viterbi(emissions[start:stop], backend)
            start = stop
        return labels

    def _posterior_sequences(
        self,
        emissions: np.ndarray,
        sequence_lengths: list[int],
        backend: _ARHMMBackend,
    ) -> np.ndarray:
        posterior = np.empty((emissions.shape[0], self.config.num_states), dtype=np.float64)
        start = 0
        for length in sequence_lengths:
            stop = start + length
            posterior[start:stop] = self._forward_backward(emissions[start:stop], backend)
            start = stop
        return posterior

    def _sequence_log_likelihood(
        self,
        emissions: np.ndarray,
        sequence_lengths: list[int],
        backend: _ARHMMBackend,
    ) -> float:
        total = 0.0
        start = 0
        for length in sequence_lengths:
            stop = start + length
            alpha = self._forward(emissions[start:stop], backend)
            total += float(self._logsumexp(alpha[-1]))
            start = stop
        return total

    def _viterbi(self, emissions: np.ndarray, backend: _ARHMMBackend) -> np.ndarray:
        n_steps, n_states = emissions.shape
        scores = np.empty((n_steps, n_states), dtype=np.float64)
        backpointers = np.zeros((n_steps, n_states), dtype=np.int64)
        scores[0] = backend.start_logprob + emissions[0]

        for step in range(1, n_steps):
            transition_scores = scores[step - 1][:, None] + backend.transition_logprob
            backpointers[step] = np.argmax(transition_scores, axis=0)
            scores[step] = transition_scores[backpointers[step], np.arange(n_states)]
            scores[step] += emissions[step]

        path = np.empty(n_steps, dtype=np.int64)
        path[-1] = int(np.argmax(scores[-1]))
        for step in range(n_steps - 2, -1, -1):
            path[step] = backpointers[step + 1, path[step + 1]]
        return path

    def _forward_backward(
        self,
        emissions: np.ndarray,
        backend: _ARHMMBackend,
    ) -> np.ndarray:
        alpha = self._forward(emissions, backend)
        beta = self._backward(emissions, backend)
        gamma = alpha + beta
        gamma -= self._logsumexp(gamma, axis=1, keepdims=True)
        return np.exp(gamma)

    def _forward(self, emissions: np.ndarray, backend: _ARHMMBackend) -> np.ndarray:
        alpha = np.empty_like(emissions)
        alpha[0] = backend.start_logprob + emissions[0]
        for step in range(1, emissions.shape[0]):
            alpha[step] = emissions[step] + self._logsumexp(
                alpha[step - 1][:, None] + backend.transition_logprob,
                axis=0,
            )
        return alpha

    def _backward(self, emissions: np.ndarray, backend: _ARHMMBackend) -> np.ndarray:
        beta = np.zeros_like(emissions)
        for step in range(emissions.shape[0] - 2, -1, -1):
            beta[step] = self._logsumexp(
                backend.transition_logprob + emissions[step + 1][None, :] + beta[step + 1][None, :],
                axis=1,
            )
        return beta

    def _build_ar_design_matrix(
        self,
        pca_trajectory: np.ndarray,
        sequence_lengths: list[int] | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        lengths = self._validate_sequence_lengths(pca_trajectory, sequence_lengths)
        ar_order = self.config.ar_order
        latent_dim = pca_trajectory.shape[1]
        valid_mask = np.zeros(pca_trajectory.shape[0], dtype=bool)
        design_rows: list[np.ndarray] = []
        target_rows: list[np.ndarray] = []

        start = 0
        for length in lengths:
            stop = start + length
            for index in range(start + ar_order, stop):
                lags = [
                    pca_trajectory[index - lag]
                    for lag in range(1, ar_order + 1)
                ]
                design_rows.append(np.concatenate(lags, axis=0))
                target_rows.append(pca_trajectory[index])
                valid_mask[index] = True
            start = stop

        if not target_rows:
            raise ValueError("pca_trajectory must contain at least one valid AR target frame")

        design = np.asarray(design_rows, dtype=np.float64).reshape(-1, latent_dim * ar_order)
        target = np.asarray(target_rows, dtype=np.float64).reshape(-1, latent_dim)
        return design, target, valid_mask

    def _expand_labels(
        self,
        valid_labels: np.ndarray,
        valid_mask: np.ndarray,
        sequence_lengths: list[int],
    ) -> np.ndarray:
        full = np.empty(valid_mask.shape[0], dtype=np.int64)
        full[valid_mask] = valid_labels

        ar_order = self.config.ar_order
        offset = 0
        start = 0
        for length in sequence_lengths:
            stop = start + length
            warmup_stop = min(stop, start + ar_order)
            if offset < valid_labels.size and stop - start > ar_order:
                fill_value = int(valid_labels[offset])
            else:
                fill_value = 0
            full[start:warmup_stop] = fill_value
            offset += max(length - ar_order, 0)
            start = stop
        return full

    def _expand_probabilities(
        self,
        valid_probabilities: np.ndarray,
        valid_mask: np.ndarray,
        sequence_lengths: list[int],
    ) -> np.ndarray:
        full = np.zeros((valid_mask.shape[0], self.config.num_states), dtype=np.float64)
        full[valid_mask] = valid_probabilities

        ar_order = self.config.ar_order
        offset = 0
        start = 0
        for length in sequence_lengths:
            stop = start + length
            warmup_stop = min(stop, start + ar_order)
            if offset < valid_probabilities.shape[0] and stop - start > ar_order:
                fill_value = valid_probabilities[offset]
            else:
                fill_value = np.full(self.config.num_states, 1.0 / self.config.num_states)
            full[start:warmup_stop] = fill_value
            offset += max(length - ar_order, 0)
            start = stop
        return full

    @staticmethod
    def _logsumexp(
        values: np.ndarray,
        axis: int | None = None,
        keepdims: bool = False,
    ) -> np.ndarray:
        max_values = np.max(values, axis=axis, keepdims=True)
        stabilized = np.exp(values - max_values)
        summed = np.sum(stabilized, axis=axis, keepdims=True)
        result = max_values + np.log(summed)
        if axis is None:
            return result if keepdims else np.asarray(result).reshape(())
        if not keepdims and axis is not None:
            result = np.squeeze(result, axis=axis)
        return result

    def _validate_trajectory(self, pca_trajectory: Any) -> np.ndarray:
        trajectory = np.asarray(pca_trajectory, dtype=np.float64)
        if trajectory.ndim != 2:
            raise ValueError("pca_trajectory must have shape (num_frames, latent_dim)")
        if trajectory.shape[0] == 0:
            raise ValueError("pca_trajectory must contain at least one frame")
        if self.config.latent_dim is not None and trajectory.shape[1] != self.config.latent_dim:
            raise ValueError("pca_trajectory latent_dim does not match config.latent_dim")
        if trajectory.shape[0] < max(self.config.num_states, self.config.ar_order + 1):
            raise ValueError("pca_trajectory must have enough frames for the configured AR-HMM")
        return trajectory

    def _validate_sequence_lengths(
        self,
        pca_trajectory: np.ndarray,
        sequence_lengths: list[int] | None,
    ) -> list[int]:
        lengths = sequence_lengths or [pca_trajectory.shape[0]]
        if sum(lengths) != pca_trajectory.shape[0]:
            raise ValueError("sequence_lengths must sum to the number of frames")
        if any(length <= self.config.ar_order for length in lengths):
            raise ValueError("each sequence length must be greater than ar_order")
        if sum(length - self.config.ar_order for length in lengths) < self.config.num_states:
            raise ValueError("pca_trajectory must contain enough valid AR frames for num_states")
        return lengths

    def _valid_lengths(self, sequence_lengths: list[int]) -> list[int]:
        return [length - self.config.ar_order for length in sequence_lengths]
