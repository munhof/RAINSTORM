import pytest


def test_native_default_architecture_preserves_existing_weights_and_outputs():
    torch = pytest.importorskip('torch')
    from rainstorm_thesis.vame_native_runtime import NativeVAMENetwork
    torch.manual_seed(42)
    baseline = NativeVAMENetwork(4, 3, 2, 5)
    torch.manual_seed(42)
    described = NativeVAMENetwork(4, 3, 2, 5, architecture={})
    values = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    assert baseline.state_dict().keys() == described.state_dict().keys()
    assert torch.equal(baseline(values, sample=False)[0], described(values, sample=False)[0])


def test_native_architecture_controls_train_and_infer_on_cpu():
    pytest.importorskip('torch')
    import numpy as np
    from rainstorm_thesis.vame_models import VAMENativeModel
    model = VAMENativeModel({'n_states': 2, 'latent_dim': 2, 'hidden_dim': 4,
        'epochs': 1, 'batch_size': 2, 'device': 'cpu', 'num_threads': 1,
        'architecture': {'encoder_layers': 2, 'encoder_bidirectional': False,
                         'decoder_layers': 2, 'decoder_hidden_dim': 3},
        'optimizer': {'type': 'adamw', 'weight_decay': .01},
        'discretizer': {'n_init': 2, 'max_iter': 5}})
    values = np.random.default_rng(42).normal(size=(6, 3, 4)).tolist()
    assert len(model.fit_predict(values).predictions) == 6
    assert 'encoder.weight_ih_l1' in model.checkpoint['model_state']
    assert 'encoder.weight_ih_l0_reverse' not in model.checkpoint['model_state']
    assert model.checkpoint['model_state']['output.weight'].shape == (4, 3)
    assert len(model.predict(values).predictions) == 6
