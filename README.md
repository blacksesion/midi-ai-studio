# MIDI-AI Studio

Reproduce archivos **MIDI** con una calidad **casi real** usando inteligencia
artificial. El programa analiza la partitura MIDI, la "interpreta" con un
modelo de expresión aprendido de pianistas reales (dataset **MAESTRO**), la
sintetiza con un soundfont de alta calidad y le aplica una mezcla/masterización
profesional (panorámica, ecualización, reverberación por convolución,
compresión, saturación y limitador).

## Qué hace (pipeline)

```
.mid ──► parseo (tempo, pistas, notas) ──► modelo de interpretación IA ──►
         síntesis por muestras (SoundFont) ──► mezcla + masterización ──► .wav
```

1. **Parseo MIDI** — lee tempo, compases, pistas, programas GM, notas y
   controladores (sustain, volumen, panorama).
2. **Interpretación expresiva (IA)** — una red neuronal predice, nota a nota,
   cómo la tocaría un intérprete humano: dinámica (velocidad), micro-timing
   y articulación (legato/staccato). El resultado se **mezcla** con la
   interpretación original mediante el parámetro `ai-strength`.
3. **Síntesis** — renderiza cada pista con *MuseScore General* (SoundFont de
   ~190 instrumentos GM) con afinación precisa por muestra.
4. **Mezcla/masterización** — coloca cada instrumento en el "escenario"
   (panorámica, nivel, envío de reverb por familia), aplica reverberación de
   convolución y una cadena de mastering.

## Requisitos

- Python 3.10 o 3.11 (recomendado 3.11)
- (Opcional) GPU NVIDIA + CUDA para entrenar el modelo; la inferencia funciona
  también en CPU.

```bash
pip install -r requirements.txt
```

El SoundFont y el dataset se descargan automáticamente la primera vez
(~38 MB + ~56 MB). El modelo incluido ya está entrenado
(`assets/models/performance.pt`); se puede reentrenar con `train.py`.

## Uso

### Línea de comandos

```bash
python -m midi_ai cancion.mid -o cancion.wav
```

Opciones principales:

```
--ai-strength 0.6   Intensidad de la interpretación IA (0..1, 0 = desactivada)
--no-ai             Desactiva el modelo de IA
--reverb 0.30       Cantidad de reverberación (0..1)
--brightness 0      Ecualizador de agudos (dB)
--warmth 0          Ecualizador de graves (dB)
--width 0           Amplitud estéreo (0..1)
--loudness -15      Volumen objetivo RMS (dBFS)
--saturation 0.5    Saturación tipo cinta (dB)
--rate 44100        Frecuencia de muestreo
--seed 42           Semilla aleatoria
```

### Interfaz web

```bash
python -m midi_ai.web
```

Se abre en `http://127.0.0.1:8765`. Arrastra un `.mid`, ajusta los controles y
pulsa **Renderizar** para escuchar y descargar el resultado en el navegador.

## Entrenar el modelo (opcional)

El modelo de interpretación se entrena sobre las interpretaciones de piano del
dataset **MAESTRO v3.0.0** (solo MIDI, ~56 MB):

```bash
python train.py --max-samples 4000000 --epochs 25
```

Esto genera `assets/models/performance.pt`. El modelo es una red MLP pequeña
que aprende a predecir, a partir del contexto musical de cada nota
(altura, posición métrica, intervalos, polifonía, posición en la frase, etc.),
la dinámica, el micro-timing y la articulación que un pianista aplicaría.

## Estructura

```
midi_ai/
  midi_parse.py    # parser de archivos MIDI
  features.py      # extracción de features musicales
  model.py         # red neuronal de interpretación expresiva
  render.py        # síntesis por muestras (TinySoundFont)
  mix.py           # mezcla y masterización
  dsp.py           # filtros, compresor, reverb, limitador
  pipeline.py      # orquesta todo el flujo
  cli.py / web.py  # interfaces
train.py           # entrenamiento del modelo
assets/
  models/          # checkpoint del modelo
  soundfonts/      # SoundFont de alta calidad
  training/        # dataset MAESTRO (para reentrenar)
```

## Créditos / licencias

- SoundFont: **MuseScore General** (MuseScore, CC0 / libre).
- Dataset de entrenamiento: **MAESTRO** (Google Magenta, CC BY-NC-SA 4.0).
- Síntesis: **TinySoundFont** (`tinysoundfont`).
