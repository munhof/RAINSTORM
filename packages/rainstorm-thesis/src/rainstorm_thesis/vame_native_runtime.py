"""PyTorch runtime for the native VAME worker adapter.

Kept in a separate module so Studio can register the adapter without importing
PyTorch. The encoder/decoder structure follows RAINSTORM's VAME implementation
in Tesis_Facu (GRU encoder, variational latent heads, and GRU reconstruction).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from sklearn.cluster import KMeans
import sklearn


class NativeVAMENetwork(nn.Module):
    def __init__(self, feature_dim: int, sequence_len: int, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.sequence_len = sequence_len
        self.latent_dim = latent_dim
        self.encoder = nn.GRU(feature_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.logvar = nn.Linear(hidden_dim * 2, latent_dim)
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, feature_dim)

    def encode(self, values):
        _, hidden = self.encoder(values)
        encoded = torch.cat((hidden[-2], hidden[-1]), dim=-1)
        return self.mu(encoded), self.logvar(encoded)

    def forward(self, values, *, sample: bool):
        mu, logvar = self.encode(values)
        latent = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if sample else mu
        repeated = self.latent_to_hidden(latent).unsqueeze(1).expand(
            -1, self.sequence_len, -1
        )
        decoded, _ = self.decoder(repeated)
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


def train_native_vame(inputs, config: dict, *, resume_state=None, checkpoint=None) -> dict:
    values = np.asarray(inputs, dtype=np.float32)
    if values.ndim != 3 or not values.shape[0] or not values.shape[1] or not values.shape[2]:
        raise ValueError("Native VAME expects (observations, window, features) pose windows")
    if not np.isfinite(values).all():
        raise ValueError("Native VAME input contains missing or non-finite pose values")
    if config["n_states"] > len(values):
        raise ValueError("n_states cannot exceed the number of training windows")
    device = torch.device(config["device"])
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    torch.set_num_threads(max(1, int(config.get("num_threads", 2))))
    model = NativeVAMENetwork(
        feature_dim=values.shape[2], sequence_len=values.shape[1],
        latent_dim=config["latent_dim"], hidden_dim=config["hidden_dim"],
    ).to(device)
    tensor = torch.as_tensor(values, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    rng = np.random.default_rng(config["seed"])
    input_spec = {"feature_dim": int(values.shape[2]),
                  "sequence_len": int(values.shape[1])}
    training_config = dict(config)
    history = []
    first_epoch = 0
    if resume_state is not None:
        if (resume_state.get("format_version") != 1
                or resume_state.get("input_spec") != input_spec
                or resume_state.get("training_config") != training_config
                or resume_state.get("torch_version") != torch.__version__
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
        optimizer.load_state_dict(resume_state["optimizer_state"])
        rng.bit_generator.state = resume_state["numpy_rng_state"]
        history = list(resume_state["training_history"])
        torch.set_rng_state(torch.as_tensor(
            resume_state["torch_rng_state"], dtype=torch.uint8, device="cpu"
        ))
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([
                torch.as_tensor(state, dtype=torch.uint8, device="cpu")
                for state in resume_state.get("cuda_rng_state", [])
            ])

    model.train()
    for epoch in range(first_epoch, config["epochs"]):
        order = rng.permutation(len(values))
        totals = []
        for start in range(0, len(order), config["batch_size"]):
            batch = tensor[order[start:start + config["batch_size"]]]
            optimizer.zero_grad(set_to_none=True)
            reconstruction, mu, logvar = model(batch, sample=True)
            reconstruction_loss = torch.nn.functional.mse_loss(reconstruction, batch)
            kld_loss = -0.5 * torch.mean(torch.sum(
                1 + logvar - mu.square() - logvar.exp(), dim=1
            ))
            loss = reconstruction_loss + float(config["kld_weight"]) * kld_loss
            loss.backward()
            optimizer.step()
            totals.append((float(loss.detach().cpu()),
                           float(reconstruction_loss.detach().cpu()),
                           float(kld_loss.detach().cpu())))
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
    with torch.no_grad():
        embeddings = model.encode(tensor)[0].cpu().numpy()
    clusterer = KMeans(n_clusters=config["n_states"], random_state=config["seed"], n_init=10)
    labels = clusterer.fit_predict(embeddings)
    return {
        "labels": labels.astype(int).tolist(),
        "embeddings": embeddings.astype(float).tolist(),
        "model_state": {name: value.detach().cpu().numpy()
                        for name, value in model.state_dict().items()},
        "cluster_centers": clusterer.cluster_centers_.astype(float),
        "input_spec": input_spec,
        "config": {key: config[key] for key in
                   ("latent_dim", "hidden_dim", "device")},
        "training_history": history,
        "seed": int(config["seed"]),
    }


def predict_native_vame(inputs, checkpoint: dict):
    values = np.asarray(inputs, dtype=np.float32)
    expected = checkpoint["input_spec"]
    if values.ndim != 3 or values.shape[1:] != (
        expected["sequence_len"], expected["feature_dim"]
    ):
        raise ValueError("Prediction windows do not match the fitted native VAME model")
    if not np.isfinite(values).all():
        raise ValueError("Prediction input contains missing or non-finite pose values")
    config = checkpoint["config"]
    device = torch.device(config["device"])
    model = NativeVAMENetwork(
        expected["feature_dim"], expected["sequence_len"],
        config["latent_dim"], config["hidden_dim"],
    ).to(device)
    state = {name: torch.as_tensor(value) for name, value in checkpoint["model_state"].items()}
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        embeddings = model.encode(torch.as_tensor(values, device=device))[0].cpu().numpy()
    centers = np.asarray(checkpoint["cluster_centers"], dtype=float)
    labels = np.square(embeddings[:, None, :] - centers[None, :, :]).sum(axis=2).argmin(axis=1)
    return labels.astype(int).tolist(), embeddings.astype(float).tolist()
