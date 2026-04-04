FROM ghcr.io/nvidia/cudaqx:0.4.0

USER root

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

ARG TORCH_INDEX_URL=""

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    make \
    git \
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install numpy scipy mpi4py

RUN git clone https://github.com/theochem/pyci.git /tmp/pyci && \
    cd /tmp/pyci && \
    make && \
    python3 -m pip install . && \
    rm -rf /tmp/pyci

WORKDIR /workspace
COPY . .

RUN if [ -n "$TORCH_INDEX_URL" ]; then \
        python3 -m pip install --extra-index-url "$TORCH_INDEX_URL" -e .; \
    else \
        python3 -m pip install -e .; \
    fi

CMD ["/bin/bash"]