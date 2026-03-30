# Use CUDA-QX v0.4.0 as the base image
FROM ghcr.io/nvidia/cudaqx:0.4.0

USER root
ENV DEBIAN_FRONTEND=noninteractive

# Python 3.11 + pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip 

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
RUN python -m pip install --upgrade pip

# Set the working directory
WORKDIR /workspace

COPY pyproject.toml .
RUN pip install -e .