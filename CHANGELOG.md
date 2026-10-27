# Changelog

## v2.0.0 (in progress)
- Feat: sequential CNN expression model (`--seq`), trained on 3.6M MAESTRO notes
  (val MSE 0.67085, MAE v_z/onset/dur = 0.590/0.049/0.287). Default checkpoint
  is now the seq model; MLP kept as `performance_mlp_backup.pt`.
- Feat: web async jobs with progress polling; family presets; multiband/de-esser;
  cached renders; parallel stem rendering.
- Feat: `scripts/evaluate_models.py` A/B evaluation (LUFS/peak/duration).
- Fix: per-note program assignment; variable tempo support; pending notes; overlapping same-pitch notes.
- Fix: train/inference feature mismatch via quantization.
- Fix: apply per-instrument brightness; keep full reverb tail; event ordering in render.
- Feat: family-aware expression shaping (drums, strings, winds, bass).
- Feat: separate timing/dynamics/articulation strengths.
- Feat: LUFS normalisation, true-peak limiter, multiformat export.
- Feat: thread-safe model cache.
- Fix: server stability — bounded render queue (1 worker), sequential stem
  rendering on <4 CPUs, per-thread synth reuse, torch thread cap, upload/body
  limits, output/cache pruning, `/healthz` and container healthcheck.
- Ops: EC2 instance resized from t3.micro (1 GB) to t3.small (2 GB) after a
  memory-exhaustion hang caused by concurrent renders. The container is capped
  at 1400 MiB (2 GB host swap), web renders are limited to 4 minutes, and a
  69 s render now peaks at ~826 MiB (was ~1.03 GiB).
- Fix: `master_loudness()` iterates gain + limiter so the LUFS target is met
  (sample: −15.03 LUFS, peak −1.01 dBFS).
- Tests: smoke suite + CI workflow.

## v0.1.0
- Initial MVP.