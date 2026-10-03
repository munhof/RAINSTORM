# Ported from Tesis_Facu; copyright Santiago D'hers. See THESIS_LICENSE.
from dataclasses import dataclass
import warnings



@dataclass(frozen=True)
class KeypointMoseqConfig:
    """KPMS model hyperparameters. No preprocessing choices."""

    num_states: int = 20
    latent_dim: int | None = None
    pca_variance_threshold: float | None = 0.90
    whiten_pca: bool = True
    ar_order: int = 3
    kappa: float = 100.0
    target_duration_seconds: float | None = None
    fps: float = 30.0
    num_iters: int = 50
    confidence_threshold: float = 0.6
    rare_state_threshold: float = 0.005
    sticky_alpha: float = 1.0
    use_sticky_prior: bool = False
    seed: int = 156
    show_progress: bool = False

    def __post_init__(self) -> None:
        if self.latent_dim is not None and self.pca_variance_threshold is not None:
            warnings.warn(
                "Both latent_dim and pca_variance_threshold were provided; "
                "pca_variance_threshold takes precedence.",
                UserWarning,
                stacklevel=2,
            )
        if self.latent_dim is None and self.pca_variance_threshold is None:
            object.__setattr__(self, "pca_variance_threshold", 0.90)
        if self.num_states < 2:
            raise ValueError("num_states must be >= 2")
        if self.ar_order < 1:
            raise ValueError("ar_order must be >= 1")
        if self.kappa < 0:
            raise ValueError("kappa must be >= 0")
        if self.sticky_alpha < 0:
            raise ValueError("sticky_alpha must be >= 0")
        if not 0 < self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be in (0, 1]")
        if not 0 < self.rare_state_threshold < 1:
            raise ValueError("rare_state_threshold must be in (0, 1)")
        if self.fps <= 0:
            raise ValueError("fps must be > 0")
        if self.num_iters < 1:
            raise ValueError("num_iters must be >= 1")

    @property
    def hyperparameters(self) -> dict:
        """Hyperparameters for the Keypoint-MoSeq model."""
        return {
            "num_states": self.num_states,
            "latent_dim": self.latent_dim,
            "pca_variance_threshold": self.pca_variance_threshold,
            "whiten_pca": self.whiten_pca,
            "ar_order": self.ar_order,
            "kappa": self.kappa,
            "target_duration_seconds": self.target_duration_seconds,
            "fps": self.fps,
            "num_iters": self.num_iters,
            "confidence_threshold": self.confidence_threshold,
            "rare_state_threshold": self.rare_state_threshold,
            "sticky_alpha": self.sticky_alpha,
            "use_sticky_prior": self.use_sticky_prior,
            "seed": self.seed,
            "show_progress": self.show_progress,
        }
