"""Load an imported model and atomically publish its actual probabilities."""
import argparse
import os
from pathlib import Path
import tempfile

# Bound CPU threads so a worker does not consume every core on a shared machine.
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '1')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
import numpy as np
import tensorflow as tf


def predict(model_path, inputs):
    model = tf.keras.models.load_model(model_path, compile=False)
    values = np.asarray(inputs, dtype=np.float32)
    if tuple(model.input_shape[1:]) not in ((12,), (7,12)):
        raise ValueError('Unsupported imported model input shape')
    if tuple(values.shape[1:]) != tuple(model.input_shape[1:]) or not np.isfinite(values).all():
        raise ValueError('Inputs disagree with imported model signature')
    if tuple(model.output_shape[1:]) != (1,):
        raise ValueError('Expected a binary model')
    return model.predict(values, verbose=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    probabilities = predict(args.model, np.load(args.inputs, allow_pickle=False))
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError('Invalid probability output')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(dir=args.output.parent, delete=False) as handle:
            name = handle.name
            np.save(handle, probabilities, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, args.output)
        directory = os.open(args.output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


if __name__ == '__main__':
    main()
