# MIDI-AI Studio — runtime image (CPU)
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MIDI_AI_HOST=0.0.0.0 \
    MIDI_AI_PORT=8765

WORKDIR /app

# libsndfile (for soundfile) + curl (to fetch the SoundFont at build time)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch first (much smaller than the CUDA build), then the rest.
# Versions match what the project is developed against.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.5.1
RUN pip install --no-cache-dir numpy==1.26.4 scipy==1.16.3 mido soundfile
# tinysoundfont lists pyaudio as a hard dep, but we only use its offline Synth
# (never its pyaudio-based playback), so skip pyaudio to avoid a source build.
RUN pip install --no-cache-dir --no-deps tinysoundfont

# Application code + the pre-trained model checkpoint.
COPY midi_ai ./midi_ai
COPY assets ./assets

# Pre-download the high-quality SoundFont so the first render is instant.
RUN mkdir -p /app/assets/soundfonts \
    && curl -fsSL -o /app/assets/soundfonts/MuseScore_General.sf3 \
       https://ftp.osuosl.org/pub/musescore/soundfont/MuseScore_General/MuseScore_General.sf3

EXPOSE 8765

CMD ["python", "-m", "midi_ai.web"]
