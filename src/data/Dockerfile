FROM python:3.12-slim

# ARG CACHE_BUSTER=0
ENV GLIBC_TUNABLES=glibc.rtld.execstack=2

ENV HF_HOME=/models \
    TRANSFORMERS_CACHE=/models \
    CTRANSATE2_CACHE_DIR=/models \
    MPLBACKEND=Agg \
    PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY data/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

RUN python - <<'PY'
import whisperx, os
print("Pre-downloading model to", os.environ.get("HF_HOME", "/root/.cache"))
_ = whisperx.load_model("medium", device="cpu", compute_type="float32")

for lang in ["en", "hi"]:  # e.g., "en","fr","hi","es"
    whisperx.load_align_model(language_code=lang, device="cpu")
PY

COPY data/data.py data/main.py data/audio.py /app/

COPY services/ /app/services/

ENTRYPOINT [ "python", "main.py" ]
