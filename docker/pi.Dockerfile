# Raspberry Pi 4 runtime container for TFLite
FROM python:3.11-slim

WORKDIR /workspace
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-pi.txt /workspace/requirements-pi.txt
RUN python -m pip install -U pip && \
    pip install -r requirements-pi.txt

COPY . /workspace
ENV PYTHONPATH=/workspace

CMD ["bash", "-lc", "python bench.py --help"]
