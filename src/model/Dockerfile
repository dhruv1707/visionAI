FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY data/requirements.txt ./requirements.txt

RUN apt-get update && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir -r requirements.txt

COPY model/main.py /app/

COPY services/ /app/services/

ENTRYPOINT [ "python", "main.py" ]