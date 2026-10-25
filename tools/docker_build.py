#!/usr/bin/env python3
"""Compile Project Maya's engine (and, on sm_70+, the vision encoder) inside the Docker build - no GPU needed.

Same flow as upstream's tools step + build step + vision step (maya.py), driven by
the CUDA_ARCHITECTURES build-arg: llama.cpp's pinned source, then CMake + Ninja for
the listed sm_ versions.  The engine needs no GPU to COMPILE - only to run.

Pascal (Tesla P40, sm_61) is built differently from the default CMP 170HX
(sm_80): upstream's CMakeLists.txt refuses any explicit arch below 7.5 and
maya.arch_setting() answers "native" (auto-detect the build machine's GPU)
there - neither works on this GPU-less build.  So the explicit arch is forced
and upstream's floor is relaxed to Maya's own minimum (Pascal, 6.0), which also
lets its `-Xptxas -dlcm=ca` path (L1-cached global loads Pascal needs) run.
Pre-Volta cards also allow only 4 KB of kernel arguments, which the mixed EXL3
kernel's exceed, so its argument list is trimmed for those builds.

Maya also has to be new enough: the sm_6x kernel guards and the `-dlcm=ca` flag
landed in v1.1.14 (PR #214), so an older ref cannot be compiled for a P40 at
all.  This build refuses anything below that rather than failing deep in nvcc.
"""
import inspect
import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import maya as M   # noqa: E402
import setup as S  # noqa: E402

# The first Maya release that supports Pascal (sm_6x): v1.1.14 added the kernel
# guards and the -dlcm=ca build flag (PR #214).
MIN_VERSION = (1, 1, 14)


def require_modern_maya() -> None:
    """Refuse a Maya older than MIN_VERSION.  We only ever ship recent builds, and anything below v1.1.14 does not
    compile for the Tesla P40 at all - name that plainly instead of letting nvcc fail on the missing guards."""
    raw = M.maya_version().lstrip("v")
    minimum = ".".join(map(str, MIN_VERSION))
    m = re.match(r"^(\d+)(?:\.(\d+))?(?:\.(\d+))?$", raw)
    if not m:
        sys.exit(f"cannot read Maya's version from VERSION ({M.maya_version()!r}) - "
                 f"pin MAYA_REF to a {minimum}+ release tag")
    found = tuple(int(x) if x else 0 for x in m.groups())
    if found < MIN_VERSION:
        sys.exit(f"Maya {raw} is too old: this image needs {minimum}+ "
                 "(Pascal support, PR #214) - pin MAYA_REF to a newer tag")


def allow_pascal_archs() -> None:
    """Relax upstream CMakeLists.txt's explicit-arch floor (sm_75) to Maya's own minimum (sm_60), so an explicit
    sm_61 - what a P40 needs - configures and picks up the `-Xptxas -dlcm=ca` flag it already sets for sm_6x.
    The patch lives in the image, so the source hash maya.py recomputes at container start matches this build's."""
    path = ROOT / "CMakeLists.txt"
    text = path.read_text(encoding="utf-8")
    patched = text.replace("if(_base LESS 75)", "if(_base LESS 60)", 1)
    if patched == text:
        sys.exit("CMakeLists.txt: expected 'if(_base LESS 75)' not found (upstream changed)")
    path.write_text(patched, encoding="utf-8")


def trim_exl3_mix_args() -> None:
    """Pre-Volta allows 4 KB of kernel arguments (32 KB from sm_70 on): set_mix_kernel takes SetArgs (4016 bytes),
    KMix (1768) and its pointers - 5828 - so nvcc refuses it for sm_61 ("Formal parameter space overflowed").  The
    kernel reads six of SetArgs' fields, so pass just those: the same code, ~1.9 KB of arguments.  (The mixed EXL3
    kernels run on the cards EXL3 supports - V100 and newer - but every arch is compiled.)  The patch lives in the
    image, so the source hash maya.py recomputes at container start matches this build's."""
    path = ROOT / "src/kernels/cuda/exl3.cu"
    if not path.exists():
        return          # (EXL3's kernels landed after v1.1.14: nothing over the limit yet)
    text = path.read_text(encoding="utf-8")
    if "set_mix_kernel(SetArgs a, KMix mx" not in text:
        return          # (no mixed-layout kernel, or upstream already trimmed it)
    edits = [
        ("""// hip_set_kernel for a mixed-precision layer: expert e's blob at a.eptr[e], each part at its class's K
template <int cb, bool gate_up>
__global__ void __launch_bounds__(kThreads) set_mix_kernel(SetArgs a, KMix mx, const float* x, int ldx,
""",
         """// hip_set_kernel for a mixed-precision layer: expert e's blob at a.eptr[e], each part at its class's K
// (pre-Volta's 4 KB of kernel arguments: SetArgs' 4016 bytes + KMix's 1768 do not fit, and the kernel reads only
// these fields of SetArgs - the body's a.bounds / a.row0 / ... still find them)
struct SetMixArgs {
    const unsigned long long* eptr;
    const int* bounds;
    int row0, n_embd, n_ff;
    float limit;
};
static SetMixArgs mix_args(const SetArgs& a) {
    return {a.eptr, a.bounds, a.row0, a.n_embd, a.n_ff, a.limit};
}
template <int cb, bool gate_up>
__global__ void __launch_bounds__(kThreads) set_mix_kernel(SetMixArgs a, KMix mx, const float* x, int ldx,
"""),
        ("""                    sa, k, x, ldx, row_idx, h16, ldh, nullptr, 0);
        });
        launch_check("set_gate_up (mixed)");""",
         """                    mix_args(sa), k, x, ldx, row_idx, h16, ldh, nullptr, 0);
        });
        launch_check("set_gate_up (mixed)");"""),
        ("""                    sa, k, nullptr, 0, nullptr, const_cast<uint16_t*>(h16), ldh, y, ldy);
        });
        launch_check("set_down (mixed)");""",
         """                    mix_args(sa), k, nullptr, 0, nullptr, const_cast<uint16_t*>(h16), ldh, y, ldy);
        });
        launch_check("set_down (mixed)");"""),
    ]
    for old, new in edits:
        if old not in text:
            sys.exit(f"{path.relative_to(ROOT)}: expected set_mix_kernel text not found (upstream changed)")
        text = text.replace(old, new, 1)
    path.write_text(text, encoding="utf-8")


def main():
    raw = os.environ.get("CUDA_ARCHITECTURES", "80")
    archs = sorted({int(x) for x in raw.replace(",", ";").split(";") if x.strip()})
    if not archs:
        sys.exit("CUDA_ARCHITECTURES must name at least one architecture (e.g. 80)")

    require_modern_maya()

    nvcc = shutil.which("nvcc")
    if not nvcc:
        sys.exit("nvcc not found: the base image must be a CUDA devel image")

    # The build has no GPU: never let CMake detect the machine's own arch ("native").  Ask for the explicit sm_ list
    # instead, and for Pascal relax the CMake floor so it is accepted.  Both happen before the source hash below, so
    # the stamped hash still matches the tree the running container sees.
    if min(archs) < 75:
        allow_pascal_archs()
    if min(archs) < 70:
        trim_exl3_mix_args()
    M.arch_setting = lambda a: ";".join(str(x) for x in a)

    llama = S.get_llama_cpp()            # pinned commit, into third_party/
    src = S.source_hash(S.ENGINE_SOURCES)

    # GitHub's free runners have limited RAM: nvcc takes 2-4 GB per parallel job
    M.mem_gb = lambda: (12.0, 12.0)

    # Upstream's CUDA build leaves ggml at its default GGML_NATIVE=ON, so cmake bakes the BUILD machine's
    # instruction set into ggml (-march=native).  The image is built on a GitHub runner (AVX-512) but runs
    # on the card's host, which often is older (a Xeon E5, a desktop Ryzen): the engine then dies with
    # SIGILL ("Illegal instruction", exit 132) the moment it reaches ggml code.  STRATA_PORTABLE=ON is
    # upstream's own switch (its HIP path sets it too); GGML_NATIVE=OFF is its ggml half - an AVX2
    # baseline, with ggml still picking AVX-512 at runtime on CPUs that have it.  compile_engine and
    # compile_vision both configure through maya.cmake_steps, so one wrapper covers both.
    _cmake_steps = M.cmake_steps
    # v1.1.17 gave cmake_steps an `nvcc` (the toolkit's release decides on Windows whether a VS 2026-only
    # PC can build); an older ref - still accepted, MIN_VERSION is v1.1.14 - does not take it.
    takes_nvcc = "nvcc" in inspect.signature(_cmake_steps).parameters

    def portable_cmake_steps(conf, build, env, bat_name, vcvars=None, nvcc=None):
        conf = [*conf, "-DSTRATA_PORTABLE=ON", "-DGGML_NATIVE=OFF"]
        if takes_nvcc:
            return _cmake_steps(conf, build, env, bat_name, vcvars, nvcc)
        return _cmake_steps(conf, build, env, bat_name, vcvars)

    M.cmake_steps = portable_cmake_steps

    result = M.compile_engine(archs=archs, gpu_ids=[], nvcc=nvcc,
                              host_compiler=None, llama=llama, src=src)
    if result is None:
        sys.exit("engine compilation failed")

    # The image (vision) encoder - llama.cpp's mtmd with GLM-5.3-Flash's vision tower patched in
    # - is upstream's setup step 7; nothing else pre-builds it, so a container's first start
    # would compile it (10-20 minutes).  Build it here too and let compile_vision write
    # build-vision/MAYA-BUILD.json with the same source hash vision_step() checks at start, so
    # the start only downloads the vision files.  The patch it applies to third_party/llama.cpp
    # is not part of ENGINE_SOURCES, so the engine's own stamp above stays valid.
    # Pascal (sm_6x, the Tesla P40) images are built text only: images need the encoder's VRAM
    # beside the expert pool, which a P40 rig has little of.  The entrypoint passes --no-vision
    # on such an image, so a first start neither downloads the 1.1 GB files nor compiles one.
    if min(archs) >= 70:
        vision_src = S.source_hash(("tools/vision",))
        if not M.compile_vision({"archs": archs, "nvcc": nvcc}, result, llama, vision_src):
            sys.exit("vision encoder compilation failed")
    else:
        print(f"skipping the vision encoder: sm_{min(archs)} (Pascal) images are text only")


if __name__ == "__main__":
    main()
