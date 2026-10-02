# Verification status — 2026-10-02

This is a verification record, not a declaration of complete scientific reconstruction.

## Verified

- STORM with the RAINSTORM plugin available: 245 tests passed, including 11
  headless browser tests. Command from the STORM checkout:
  `PYTHONPATH=../RAINSTORM/packages/rainstorm-thesis/src:../RAINSTORM/src .venv/bin/pytest -q --tb=short`.
- Scientific adapter regression tests in the ROCm worker: 41 passed, 4 skipped
  with GPU availability disabled for CPU regression checks.
- Native GRU backend: 30 consecutive forward/backward batches on GPU, batch 256,
  synthetic windows of 15 frames and 30 coordinates, hidden dimension 128.
  This does not establish stability on all biological windows.
- Legacy checkpoint lacking `rnn_backend` accepts explicit `auto`; changing it
  to `native` is still rejected as a change of training recipe.
- Studio and worker deployed. Fatal GPU aborts preserve signal/last observed
  batch and clear active rate/ETA indicators. UI directs GPU-hang recovery to
  a new plan instead of an unchanged retry.
- Biological-data reconstruction submitted as study 8, plan revision 64,
  job `2aa9db52-4524-4f24-8644-b16c4c6423fe`: GPU, native GRU backend, 25 epochs.
  Its initial preparation completed and training emitted batch progress.

## Acceptance still open

- Completion and checkpoint recovery of the full native biological training.
- Final official VAME reconstruction on the registered historical partition.
- Inference from completed reconstruction bundles on the protected benchmark,
  aligned metrics and scientifically justified comparison.
- Verify supervised binary-task correspondence before any common ranking against
  the 12-category human reference. Existing inference artifacts alone do not
  prove valid classification metrics.
- Full clean-clone distribution verification and remaining coverage listed in
  the reconstruction README.

For current job state, use Studio rather than treating this dated record as live progress.
