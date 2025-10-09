# Jetson Nano runtime container with TensorRT
FROM nvcr.io/nvidia/l4t-ml:r35.4.1-py3

WORKDIR /workspace
RUN apt-get update && apt-get install -y python3-pip && rm -rf /var/lib/apt/lists/*
COPY requirements-jetson.txt /workspace/requirements-jetson.txt
RUN python3 -m pip install -U pip && \
    pip3 install -r requirements-jetson.txt --extra-index-url https://developer.download.nvidia.com/compute/redist

COPY . /workspace
ENV PYTHONPATH=/workspace

CMD ["bash", "-lc", "python3 bench.py --help"]
