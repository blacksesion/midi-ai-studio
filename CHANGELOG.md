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
- Tests: smoke suite + CI workflow.

## v0.1.0
- Initial MVP.