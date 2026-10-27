# Plan de evolución: MIDI-AI Studio v2.0 ("Next Level")

> Generado a partir del estudio completo del repositorio (v0.1.0 / MVP).
> El MVP funcionó muy bien; este plan lo lleva un nivel más allá.

## Visión "North Star"

- Audio más realista y consistente: expresión musical coherente por frase.
- Multi-instrumento real: cada familia con su propia expresión y tratamiento.
- Control fino: intensidad separada por dimensión + presets por género.
- Producto usable: web con progreso/cancelación, exportación MP3/FLAC, CLI por lotes, caché.
- Ingeniería: tests, CI, empaquetado, modelo optimizado (ONNX/int8).

## Fase 0 — Corrección y consistencia (quick wins)

Bugs reales que hoy limitan calidad. Mayor ROI.

1. Programa MIDI por nota mal asignado (`midi_parse.py`).
2. Tempo fijo: soportar tempo variable (`midi_parse.py`, `features.py`, `pipeline.py`).
3. Notas pendientes al final con duración 0 → 0.05 s.
4. Solape de misma altura: usar pila en `pending_on`.
5. Desajuste train/inferencia: cuantizar entrada a grid antes de features.
6. `bright_db` del preset no se aplica (`mix.py`).
7. Cola de reverb cortada (`mix.py`).
8. Orden de eventos en render (`render.py`).
9. Caché de modelo sin lock (`pipeline.py`).
10. Unificar flags `humanize`/`use_model` (`web.py`).

## Fase 1 — Motor de audio y mezcla

1. Mastering: LUFS, true-peak limiter (oversampling), compresor multibanda, de-esser.
2. Procesamiento por pista: HPF, EQ, compresión ligera, depth, chorus.
3. Reverb: presets (sala, hall, plate, catedral) + IR externos, tail completo.
4. Panorama orquestal por familia.
5. Dithering en 16-bit; exportar 24-bit/float.
6. Fade in/out y recorte de silencio.
7. Salida multiformato (WAV/FLAC/OGG/MP3 vía ffmpeg).

## Fase 2 — Modelo de expresión v2

1. Expresión por familia de instrumento (piano neuronal, cuerdas/vientos reglas, batería groove).
2. Modelo neuronal secuencial (CNN/Transformer) con más features y targets (pedal, expresión, mod wheel).
3. Controles separados: timing/dynamics/articulation/humanize.
4. Presets por género.
5. Modelo versionado con fallback.

## Fase 3 — Síntesis y renderizado

1. Backends intercambiables (TSF por defecto, FluidSynth opcional).
2. SoundFonts por pista.
3. Renderizado paralelo + caché de stems.
4. Micro-pitch humanization vía pitch bend.
5. Streaming/chunking + progress callback.
6. Caché de resultados por hash.

## Fase 4 — Producto (Web + CLI)

1. Web asíncrona con progreso (SSE), cancelación, historial.
2. Visualización: waveform, espectrograma, A/B.
3. Presets guardables, recent files.
4. Batch CLI con config YAML/JSON.
5. API REST documentada (OpenAPI).
6. Docker multi-stage, healthcheck, volúmenes, variante GPU.
7. I18n y accesibilidad.

## Fase 5 — Entrenamiento y datos

1. Split por pieza, métricas, early stopping, mixed precision.
2. Data augmentation.
3. Caché de features.
4. DDP opcional.
5. Export TorchScript/ONNX + cuantización int8.
6. Reporte de entrenamiento.

## Fase 6 — Calidad de ingeniería

1. Tests (pytest).
2. Lint/format (ruff, black) + mypy.
3. CI (GitHub Actions).
4. Versionado 2.0.0, CHANGELOG, docs (mkdocs).
5. Logging estructurado.
6. Errores accionables.

## Priorización

- Sprint 1: Fase 0 completa + Fase 1.1–1.3.
- Sprint 2: Fase 2.1 + controles separados + presets.
- Sprint 3: Fase 2.2 + Fase 5.
- Sprint 4: Fase 3 + Fase 4.
- Continuo: Fase 6.

## Métricas de éxito

- Loudness consistente (±0.5 LU), sin clipping, tail de reverb completo.
- Preferencia humana en A/B; timing natural ~10–40 ms; dinámica correlacionada con MAESTRO.
- Render <2× realtime en CPU.
- 0 crashes en 50 MIDIs diversos; tests en CI.

## Riesgos

- Modelo secuencial requiere reentrenar → mantener v1 como fallback.
- FluidSynth/ffmpeg dependencias opcionales con detección en runtime.
- Cambios de parser → tests de regresión + golden files.
