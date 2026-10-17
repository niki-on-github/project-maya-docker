# syntax=docker/dockerfile:1

# Project Maya (https://github.com/mw00/project-maya) in a container.
# The engine compiles once for ONE CUDA architecture (CMP 170HX = GA100 = sm_80
# by default; override CUDA_ARCHITECTURES for a different card - a Tesla P40 is
# Pascal sm_61, built differently by tools/docker_build.py).  The model
# (96-156 GB) and Maya's configs live outside the image in /models and /data
# (see docker-entrypoint.sh).
#
# Build with a specific upstream tag:
#   docker build --build-arg MAYA_REF=v1.1.15 -t maya .         # CMP 170HX, sm_80
#   docker build --build-arg CUDA_ARCHITECTURES=61 -t maya-p40 . # Tesla P40, sm_61
ARG CUDA_BASE=12.8.1-devel-ubuntu24.04
FROM nvidia/cuda:${CUDA_BASE}

ARG MAYA_REF=v1.1.15
ARG CUDA_ARCHITECTURES=80

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}
# (CUDA_ARCHITECTURES stays in the image: docker-entrypoint.sh reads it at start
# to pick the drafting default - DFlash2 on sm_80+, the MTP block otherwise.)

# git is only needed here (the upstream source is cloned in the image build); a
# CUDA *devel* image is required for nvcc at build time.
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      curl \
      git \
      libgomp1 \
      python3 \
      python3-venv \
      && rm -rf /var/lib/apt/lists/*

RUN git clone --branch "$MAYA_REF" --depth 1 \
      https://github.com/mw00/project-maya.git /opt/project-maya

WORKDIR /opt/project-maya

# Honor HF_ENDPOINT at runtime (a HuggingFace mirror, e.g. an olah cache):
# upstream hardcodes https://huggingface.co in maya.py's one download-URL
# constant (HF / hf_url) - model, vision encoder and drafter all go through it.
# The grep must succeed or the image build fails (upstream refactored the line).
RUN sed -i \
      's|^HF = "https://huggingface.co/{repo}/resolve/{revision}/{path}"$|HF = os.environ.get("HF_ENDPOINT", "https://huggingface.co") + "/{repo}/resolve/{revision}/{path}"|' \
      maya.py \
    && grep -q 'HF_ENDPOINT' maya.py

# python3-pip is absent on Ubuntu 24.04, but ensurepip (in the venv) works
RUN python3 -m venv .venv \
    && .venv/bin/pip install --upgrade pip \
    # exact upstream list (setup.py PY_PACKAGES)
    && .venv/bin/pip install --no-cache-dir \
         numpy jinja2 regex pyyaml tqdm requests cmake ninja pillow psutil \
    && chmod +x maya.sh setup.sh

# Compile the engine for the configured arch(es) and write build/MAYA-BUILD.json
# (same source-hash check maya.py runs at start, so the container start skips compiling)
COPY tools/docker_build.py tools/docker_build.py
RUN .venv/bin/python tools/docker_build.py

# Persistent home ($HOME) keeps the calibration state in /data.
# MAYA_MODEL is the quant the first start downloads; override it per container
# with -e MAYA_MODEL=... (e.g. Maya-S24 on a 24 GB card), or -e MAYA_MODEL= to
# fall back to the setup's own recommendation.
ENV HOME=/data/maya/home \
    PATH=/opt/project-maya/.venv/bin:$PATH \
    MAYA_MODEL=Maya-M-Derisked

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint
RUN chmod +x /usr/local/bin/docker-entrypoint \
    && mkdir -p /data/maya/configs /data/maya/home /models

VOLUME ["/models", "/data"]

EXPOSE 8080

ENTRYPOINT ["docker-entrypoint"]
