"""PyTorch runtime for the native VAME worker adapter.

Kept in a separate module so Studio can register the adapter without importing
PyTorch. The encoder/decoder structure follows RAINSTORM's VAME implementation
in Tesis_Facu (GRU encoder, variational latent heads, and GRU reconstruction).
"""
from __future__ import annotations

from math import ceil
from time import monotonic
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch import nn
from sklearn.cluster import KMeans
import sklearn


class NativeVAMENetwork(nn.Module):
    def __init__(self, feature_dim: int, sequence_len: int, latent_dim: int, hidden_dim: int, rnn_backend="auto"):
        super().__init__()
        if rnn_backend not in {"auto", "native"}:
            raise ValueError("rnn_backend must be auto or native")
        self.rnn_backend = rnn_backend
        self.sequence_len = sequence_len
        self.latent_dim = latent_dim
        self.encoder = nn.GRU(feature_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.logvar = nn.Linear(hidden_dim * 2, latent_dim)
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, feature_dim)

    def _run_gru(self, layer, values):
        if self.rnn_backend == "native":
            with torch.backends.cudnn.flags(enabled=False):
                return layer(values)
        return layer(values)

    def encode(self, values):
        _, hidden = self._run_gru(self.encoder, values)
        encoded = torch.cat((hidden[-2], hidden[-1]), dim=-1)
        return self.mu(encoded), self.logvar(encoded)

    def forward(self, values, *, sample: bool):
        mu, logvar = self.encode(values)
        latent = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if sample else mu
        repeated = self.latent_to_hidden(latent).unsqueeze(1).expand(
            -1, self.sequence_len, -1
        )
        decoded, _ = self._run_gru(self.decoder, repeated)
        return self.output(decoded), mu, logvar


def _cpu_state(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_state(item) for item in value)
    return value


def resolve_device(requested: str) -> torch.device:
    """Choose a supported runtime device, preferring CUDA/HIP for ``auto``."""
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "A GPU device was requested, but this worker has no available "
            "CUDA or ROCm device. Start the GPU worker or choose CPU."
        )
    return device


def _report_batch(callback, *, phase, label, completed, total, batch_size, started,
                  device, epoch=None, epochs=None, loss=None):
    if callback is None:
        return
    elapsed = max(0.0, monotonic() - started)
    speed = completed / elapsed if elapsed > 0 and completed else None
    callback({
        'phase': phase, 'label': label, 'epoch': epoch, 'device': str(device),
        'phase_step': epoch - 1 if epoch is not None else completed,
        'phase_total': epochs if epoch is not None else total,
        'unit_label': 'épocas' if epoch is not None else 'observaciones',
        'batch_step': ceil(completed / batch_size), 'batch_total': ceil(total / batch_size),
        'processed_observations': completed, 'total_observations': total,
        'batch_elapsed_seconds': elapsed, 'throughput': speed,
        'batch_eta_seconds': ceil((total - completed) / speed) if speed else None,
        'loss': loss,
    })


def train_native_vame(inputs, config: dict, *, resume_state=None, checkpoint=None,
                      progress_callback=None) -> dict:
    storage = config.get("window_storage", "memory")
    if storage not in {"memory", "mapped"}:
        raise ValueError("window_storage must be memory or mapped")
    if storage == "memory":
        return _train_native_vame_values(inputs, config, resume_state=resume_state,
                                        checkpoint=checkpoint, progress_callback=progress_callback)
    if not len(inputs):
        raise ValueError("Native VAME expects nonempty pose windows")
    first = np.asarray(inputs[0], dtype=np.float32)
    if first.ndim != 2 or not all(first.shape):
        raise ValueError("Native VAME expects (observations, window, features) pose windows")
    batch_size = max(1, int(config.get("batch_size", 256)))
    with TemporaryDirectory(prefix="rainstorm-vame-windows-") as directory:
        mapped = np.lib.format.open_memmap(Path(directory) / "windows.npy", mode="w+",
                                          dtype=np.float32, shape=(len(inputs), *first.shape))
        started = monotonic()
        info = dict(phase='materializing', label='Preparando ventanas en almacenamiento temporal',
                    total=len(inputs), batch_size=batch_size, started=started, device='cpu')
        _report_batch(progress_callback, completed=0, **info)
        for start in range(0, len(inputs), batch_size):
            end = min(start + batch_size, len(inputs))
            chunk = np.asarray(inputs[start:end], dtype=np.float32)
            if chunk.shape != (end-start, *first.shape) or not np.isfinite(chunk).all():
                raise ValueError("Native VAME windows must be rectangular and finite")
            mapped[start:end] = chunk
            _report_batch(progress_callback, completed=end, **info)
        mapped.flush()
        return _train_native_vame_values(mapped, config, resume_state=resume_state,
                                        checkpoint=checkpoint, progress_callback=progress_callback)


def _train_native_vame_values(inputs, config, *, resume_state=None, checkpoint=None,
                              progress_callback=None):
    values = np.asarray(inputs, dtype=np.float32)
    if values.ndim != 3 or not values.shape[0] or not values.shape[1] or not values.shape[2]:
        raise ValueError("Native VAME expects (observations, window, features) pose windows")
    if any(not np.isfinite(values[start:start + 4096]).all()
           for start in range(0, len(values), 4096)):
        raise ValueError("Native VAME input contains missing or non-finite pose values")
    if config["n_states"] > len(values):
        raise ValueError("n_states cannot exceed the number of training windows")
    device = resolve_device(config.get("device", "auto"))
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    torch.set_num_threads(max(1, int(config.get("num_threads", 2))))
    model = NativeVAMENetwork(
        feature_dim=values.shape[2], sequence_len=values.shape[1],
        latent_dim=config["latent_dim"], hidden_dim=config["hidden_dim"],
        rnn_backend=config.get("rnn_backend", "auto"),
    ).to(device)
    # Keep the complete source array on CPU; only the active batch moves to the device.
    tensor = torch.as_tensor(values)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    rng = np.random.default_rng(config["seed"])
    input_spec = {"feature_dim": int(values.shape[2]),
                  "sequence_len": int(values.shape[1])}
    training_config = dict(config)
    history = []
    first_epoch = 0
    if resume_state is not None:
        saved_training_config = dict(resume_state.get("training_config") or {})
        requested_training_config = dict(training_config)
        saved_training_config.setdefault("rnn_backend", "auto")
        requested_training_config.setdefault("rnn_backend", "auto")
        saved_training_config.setdefault("window_storage", "memory")
        requested_training_config.setdefault("window_storage", "memory")
        device_changed = (
            saved_training_config.get("device") != requested_training_config.get("device")
        )
        saved_training_config.pop("device", None)
        requested_training_config.pop("device", None)
        saved_epoch = resume_state.get("epoch")
        final_epoch = type(saved_epoch) is int and saved_epoch == config["epochs"]
        torch_version_matches = resume_state.get("torch_version") == torch.__version__
        torch_version_compatible = torch_version_matches or (device_changed and final_epoch)
        if (resume_state.get("format_version") != 1
                or resume_state.get("input_spec") != input_spec
                or saved_training_config != requested_training_config
                or not torch_version_compatible
                or resume_state.get("numpy_version") != np.__version__
                or resume_state.get("sklearn_version") != sklearn.__version__):
            raise ValueError("Native VAME checkpoint is incompatible with this data, recipe, or runtime")
        first_epoch = resume_state.get("epoch")
        if type(first_epoch) is not int or not 0 <= first_epoch <= config["epochs"]:
            raise ValueError("Native VAME checkpoint epoch is invalid")
        required = {"model_state", "optimizer_state", "numpy_rng_state", "torch_rng_state",
                    "training_history"}
        if not required <= set(resume_state):
            raise ValueError("Native VAME checkpoint is incomplete")
        model.load_state_dict({
            name: torch.as_tensor(value, device=device)
            for name, value in resume_state["model_state"].items()
        })
        if not final_epoch:
            optimizer.load_state_dict(resume_state["optimizer_state"])
        rng.bit_generator.state = resume_state["numpy_rng_state"]
        history = list(resume_state["training_history"])
        torch.set_rng_state(torch.as_tensor(
            resume_state["torch_rng_state"], dtype=torch.uint8, device="cpu"
        ))
        cuda_rng_state = resume_state.get("cuda_rng_state", [])
        if device.type == "cuda" and cuda_rng_state:
            torch.cuda.set_rng_state_all([
                torch.as_tensor(state, dtype=torch.uint8, device="cpu")
                for state in cuda_rng_state
            ])

    model.train()
    for epoch in range(first_epoch, config["epochs"]):
        order = rng.permutation(len(values))
        totals = []
        started = monotonic()
        batch_info = dict(phase='training', label='Entrenando VAME nativo',
                          total=len(values), batch_size=config['batch_size'],
                          started=started, device=device, epoch=epoch + 1, epochs=config['epochs'])
        _report_batch(progress_callback, completed=0, **batch_info)
        for start in range(0, len(order), config["batch_size"]):
            batch = tensor[order[start:start + config["batch_size"]]].to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction, mu, logvar = model(batch, sample=True)
            reconstruction_loss = torch.nn.functional.mse_loss(reconstruction, batch)
            kld_loss = -0.5 * torch.mean(torch.sum(
                1 + logvar - mu.square() - logvar.exp(), dim=1
            ))
            loss = reconstruction_loss + float(config["kld_weight"]) * kld_loss
            if not bool(torch.isfinite(loss).item()):
                message = f'Non-finite loss at epoch {epoch + 1}, batch {start // config["batch_size"] + 1}'
                if progress_callback is not None:
                    progress_callback({'phase': 'training', 'label': message, 'status': 'failed',
                                       'failure_kind': 'nonfinite_loss', 'epoch': epoch + 1,
                                       'batch_step': start // config['batch_size'] + 1,
                                       'batch_total': ceil(len(order) / config['batch_size']),
                                       'device': str(device), 'rnn_backend': config.get('rnn_backend', 'auto')})
                raise FloatingPointError(message)
            loss.backward()
            optimizer.step()
            totals.append((float(loss.detach().cpu()),
                           float(reconstruction_loss.detach().cpu()),
                           float(kld_loss.detach().cpu())))
            _report_batch(progress_callback,
                          completed=min(start + config['batch_size'], len(values)),
                          loss=totals[-1][0], **batch_info)
        history.append({
            "epoch": epoch + 1,
            "total_loss": float(np.mean([item[0] for item in totals])),
            "reconstruction_loss": float(np.mean([item[1] for item in totals])),
            "kld_loss": float(np.mean([item[2] for item in totals])),
        })
        if checkpoint is not None:
            state = {
                "format_version": 1,
                "epoch": epoch + 1,
                "model_state": {name: value.detach().cpu().numpy().copy()
                                for name, value in model.state_dict().items()},
                "optimizer_state": _cpu_state(optimizer.state_dict()),
                "numpy_rng_state": rng.bit_generator.state,
                "torch_rng_state": torch.get_rng_state().cpu().numpy().copy(),
                "training_history": list(history),
                "input_spec": input_spec,
                "training_config": training_config,
                "torch_version": torch.__version__,
                "numpy_version": np.__version__,
                "sklearn_version": sklearn.__version__,
            }
            if device.type == "cuda":
                state["cuda_rng_state"] = [item.cpu().numpy().copy()
                                            for item in torch.cuda.get_rng_state_all()]
            checkpoint(state)

    model.eval()
    embeddings = _encode_in_batches(model, tensor, config["batch_size"], device,
                                    progress_callback=progress_callback, phase='encoding')
    if progress_callback is not None:
        progress_callback({'phase': 'clustering', 'label': 'Ajustando el discretizador KMeans',
                           'phase_step': None, 'phase_total': None, 'unit_label': None,
                           'device': 'cpu', 'epoch': None})
    clusterer = KMeans(n_clusters=config["n_states"], random_state=config["seed"], n_init=10)
    labels = clusterer.fit_predict(embeddings)
    return {
        "labels": labels.astype(int).tolist(),
        "embeddings": embeddings,
        "model_state": {name: value.detach().cpu().numpy()
                        for name, value in model.state_dict().items()},
        "cluster_centers": clusterer.cluster_centers_.astype(float),
        "input_spec": input_spec,
        "config": {
            "latent_dim": config["latent_dim"],
            "hidden_dim": config["hidden_dim"],
            "device": str(device),
            "batch_size": config["batch_size"],
        },
        "training_history": history,
        "seed": int(config["seed"]),
    }


def predict_native_vame(inputs, checkpoint: dict, *, progress_callback=None):
    values = np.asarray(inputs, dtype=np.float32)
    expected = checkpoint["input_spec"]
    if values.ndim != 3 or values.shape[1:] != (
        expected["sequence_len"], expected["feature_dim"]
    ):
        raise ValueError("Prediction windows do not match the fitted native VAME model")
    if not np.isfinite(values).all():
        raise ValueError("Prediction input contains missing or non-finite pose values")
    config = checkpoint["config"]
    device = resolve_device(config.get("device", "cpu"))
    model = NativeVAMENetwork(
        expected["feature_dim"], expected["sequence_len"],
        config["latent_dim"], config["hidden_dim"],
        rnn_backend=config.get("rnn_backend", "auto"),
    ).to(device)
    state = {name: torch.as_tensor(value) for name, value in checkpoint["model_state"].items()}
    model.load_state_dict(state)
    model.eval()
    embeddings = _encode_in_batches(
        model, torch.as_tensor(values), int(config.get("batch_size", 256)), device,
        progress_callback=progress_callback,
    )
    centers = np.asarray(checkpoint["cluster_centers"], dtype=float)
    labels = np.square(embeddings[:, None, :] - centers[None, :, :]).sum(axis=2).argmin(axis=1)
    return labels.astype(int).tolist(), embeddings.astype(float).tolist()


def _encode_in_batches(model, values, batch_size: int, device, *,
                       progress_callback=None, phase='predicting') -> np.ndarray:
    """Encode all observations without allocating sequence activations for the full dataset."""
    if batch_size < 1:
        raise ValueError("VAME encoding batch_size must be positive")
    embeddings = np.empty((len(values), model.latent_dim), dtype=np.float32)
    started = monotonic()
    batch_info = dict(phase=phase, label='Codificando ventanas de pose',
                      total=len(values), batch_size=batch_size, started=started, device=device)
    _report_batch(progress_callback, completed=0, **batch_info)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            end = min(start + batch_size, len(values))
            batch = values[start:end].to(device)
            encoded, _ = model.encode(batch)
            embeddings[start:end] = encoded.cpu().numpy()
            _report_batch(progress_callback, completed=end, **batch_info)
    return embeddings
