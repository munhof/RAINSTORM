import numpy as np
import pytest


def test_recovered_inference_uses_saved_thread_limit(monkeypatch):
    torch = pytest.importorskip('torch')
    from rainstorm_thesis import vame_native_runtime as runtime
    config = {'device': 'cpu', 'num_threads': 2, 'latent_dim': 2, 'hidden_dim': 4, 'batch_size': 2}
    network = runtime.NativeVAMENetwork(4, 3, 2, 4)
    checkpoint = {'config': config, 'input_spec': {'sequence_len': 3, 'feature_dim': 4},
        'model_state': {name: tensor.detach().numpy() for name, tensor in network.state_dict().items()},
        'cluster_centers': [[0., 0.]]}
    seen = []
    encode = runtime._encode_in_batches
    def recording_encode(*args, **kwargs):
        seen.append(torch.get_num_threads())
        return encode(*args, **kwargs)
    monkeypatch.setattr(runtime, '_encode_in_batches', recording_encode)
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(4)
        labels, _ = runtime.predict_native_vame(np.ones((2, 3, 4), dtype=np.float32), checkpoint)
        assert labels == [0, 0]
        assert seen == [2]
    finally:
        torch.set_num_threads(previous)
