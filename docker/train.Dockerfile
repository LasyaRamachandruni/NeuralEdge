# Training and export container
FROM nvcr.io/nvidia/pytorch:24.05-py3

WORKDIR /workspace
COPY requirements.txt /workspace/requirements.txt
RUN python3 -m pip install -U pip && \
    pip install -r requirements.txt

COPY . /workspace
ENV PYTHONPATH=/workspace

CMD ["bash", "-lc", "make help"]
