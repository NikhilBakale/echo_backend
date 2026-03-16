FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Runtime/system packages needed by librosa, soundfile, OpenCV, and matplotlib.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsndfile1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip setuptools wheel && pip install -r requirements.txt

COPY . .

RUN mkdir -p static/spectrograms static/audio

ENV FLASK_ENV=production \
    PORT=8000

EXPOSE 8000

CMD ["sh", "-c", "gunicorn --bind=0.0.0.0:${PORT:-8000} --timeout ${GUNICORN_TIMEOUT:-600} --workers ${GUNICORN_WORKERS:-2} --threads ${GUNICORN_THREADS:-4} app:app"]
