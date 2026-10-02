# Verification status — 2026-10-02

This is a verification record, not a declaration of complete scientific reconstruction.

## Verified

- STORM with the RAINSTORM plugin available: 251 tests passed, including 12
  headless browser tests. Command from the STORM checkout:
  `PYTHONPATH=../RAINSTORM/packages/rainstorm-thesis/src:../RAINSTORM/src .venv/bin/pytest -q --tb=short`.
- Scientific adapter regression tests in the ROCm worker: 44 passed, 4 skipped
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
  Its initial preparation completed; training emitted one batch and then
  aborted with GPU Hang/SIGABRT. The native backend has not resolved stability
  on the real reconstruction workload. The failed run remains preserved.
- Revision 65 adds mapped window storage. Job
  `0e37f08f-4d46-49d4-8385-9e3643b2d771` completed window materialization
  but also aborted with GPU Hang/SIGABRT in epoch 1 after its first batch.
  Changing window storage has not resolved the full-workload failure.
- Bounded biological probes with one training and one evaluation session,
  8192 frames each, completed 32 training batches and encoding on GPU.
  Memory and mapped storage produced identical histories. These diagnostic
  models do not replace the final reconstruction.
- Supervised reports display aligned binary probability histograms and raw
  frequency tables. The positive class remains explicitly unmapped where its
  behavioral meaning is unknown.

## Next GPU diagnostic

The host driver logged MES REMOVE_QUEUE/SUSPEND failures, queue eviction
failures and GPU resets. A related gfx1103 report is tracked in
[ROCm issue 6386](https://github.com/ROCm/legacy-rocm-build/issues/6386).
This similarity does not establish an identical cause or a working remedy.

The host runs `7.2.6-zen2-1-zen`; `6.18.53-1-lts` is already installed with
`/boot/vmlinuz-linux-lts` and `/boot/initramfs-linux-lts.img`.
The next controlled diagnostic is a single boot into that existing LTS kernel:

1. Save desktop work and restart the host after explicit approval.
2. Select the installed Linux LTS entry in GRUB without changing its default.
3. Verify `uname -r`, worker GPU visibility and Studio availability.
4. Repeat the bounded biological probe first. Keep output and driver logs.
5. Only after that passes, run a separately identified full-data diagnostic
   preserving seed, partitions and model parameters; retain all failed jobs.

No bootloader settings have been changed. A container restart cannot switch
the host kernel. The LTS comparison is a diagnostic, not a promised fix.

## Acceptance still open

Clean committed copies of RAINSTORM `65b416b` and STORM `76e09bd` passed all
8 scaffolding tests. All COPY inputs for the CPU, CUDA and ROCm Containerfiles
exist in those copies, without Tesis_Facu. This checks source completeness;
it does not verify all deployment variants. The worker guide now specifies
the exact STORM commit for a new sibling checkout.

The CPU worker image built successfully from these clean copies as
`localhost/rainstorm-clean-verification:20261002`, image ID
`af157c986bd3607833aab1e4009f95fa6ec01fc056e58b059c893e4fe29fd25d`.
A disposable container imported STORM, the RAINSTORM plugin, PyTorch
`2.14.0+cpu` and TensorFlow `2.10.1`, without the study workspace mounted.
The supervised requirements lock also installed successfully there. All three
Containerfiles now use that lock before installing the supervised package
without dependency resolution; its regression failed before the fix and all
9 scaffolding tests pass afterwards. The built image predates this Containerfile
fix; fresh CUDA/ROCm builds remain open.

This same image also migrated an empty workspace and started Studio on an
isolated port. Headless Chromium created a study and loaded data, preparation,
flow, models, jobs, evidence, comparison and reports with HTTP 200. No existing
study database or artifacts were mounted. This verifies initial installation
and navigation, not an end-to-end scientific experiment in that fresh workspace.

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
