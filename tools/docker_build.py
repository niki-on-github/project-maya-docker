#!/usr/bin/env python3
"""Compile Project Maya's engine inside the Docker build - no GPU needed.

Same flow as upstream's tools step + build step (maya.py), driven by the
CUDA_ARCHITECTURES build-arg: llama.cpp's pinned source, then CMake + Ninja for
the listed sm_ versions.  The engine needs no GPU to COMPILE - only to run.
"""
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import maya as M   # noqa: E402
import setup as S  # noqa: E402


def main():
    raw = os.environ.get("CUDA_ARCHITECTURES", "80")
    archs = sorted({int(x) for x in raw.replace(",", ";").split(";") if x.strip()})
    if not archs:
        sys.exit("CUDA_ARCHITECTURES must name at least one architecture (e.g. 80)")

    nvcc = shutil.which("nvcc")
    if not nvcc:
        sys.exit("nvcc not found: the base image must be a CUDA devel image")

    llama = S.get_llama_cpp()            # pinned commit, into third_party/
    src = S.source_hash(S.ENGINE_SOURCES)

    # GitHub's free runners have limited RAM: nvcc takes 2-4 GB per parallel job
    M.mem_gb = lambda: (12.0, 12.0)

    result = M.compile_engine(archs=archs, gpu_ids=[], nvcc=nvcc,
                              host_compiler=None, llama=llama, src=src)
    if result is None:
        sys.exit("engine compilation failed")


if __name__ == "__main__":
    main()
