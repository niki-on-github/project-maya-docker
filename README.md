Project Maya in Docker — [mw00/project-maya](https://github.com/mw00/project-maya)
(GLM-5.3-Flash, 321 B MoE, MIT), built for **one CUDA architecture per image**.
Two are published: `sm80` for the **NVIDIA CMP 170HX / A100** (GA100, compute
capability 8.0) and `sm61` for the **Tesla P40 / P100** (Pascal, 6.1). The model
lives outside the image and downloads on first start.

## CI (GitHub Actions)

A daily cron (this repo, "Build and publish Docker image") resolves upstream's
newest tag and builds and pushes **both** architectures to GHCR as
`ghcr.io/<owner>/<repo>:<arch>-<upstream-tag>` (e.g. `sm80-v1.1.15`,
`sm61-v1.1.15`) and `:<arch>-latest`.  Manual runs and pushes to `main` build
too; every run rebuilds both architectures.

| Image | `<arch>` | GPU | `CUDA_ARCHITECTURES` | Default drafting | Vision |
|---|---|---|---|---|---|
| CMP 170HX / A100 | `sm80` | compute capability 8.0 | `80` | DFlash2 | yes |
| Tesla P40 / P100 | `sm61` | Pascal, compute capability 6.1 | `61` | MTP block | no (text only) |

Build input (`workflow_dispatch`):

| Input | Default | Meaning |
|---|---|---|
| `maya_ref` | newest upstream tag | upstream ref/tag to build |

On a GitHub-hosted runner each build drops from ~37 min for all seven upstream
architectures (cold, no cache) to roughly **10–15 min** for the single-arch
engine, plus another ~10–20 min the first time for the vision encoder (built on
sm_70+ only — the `sm61` image is text only; both are built in the image, so a
container's first start only downloads).  With the GHA
layer cache warm it is faster.  The P40 (`sm61`) image is
built differently — see [Rebuild for a different NVIDIA card](#rebuild-for-a-different-nvidia-card).

## Run

You need the NVIDIA Container Toolkit (`nvidia-container-toolkit`) and a GPU
that matches the image: compute capability 8.0 for `sm80` (A100, CMP 170HX,
GA100 mining cards), 6.1 for `sm61` (Tesla P40/P100).  On first start the image
downloads **Maya-M-Derisked** (116 GB) unless you set `MAYA_MODEL`.

### CMP 170HX (`sm80`)

```sh
mkdir -p /path/on/nvme/models          # ~97-156 GB, on a fast NVMe SSD
docker run -d --name maya-170hx \
  --gpus all \
  --ipc host \
  --cap-add IPC_LOCK \
  --ulimit memlock=-1 \
  -p 8080:8080 \
  -v /path/on/nvme/models:/models \
  -v /path/to/configs:/data \
  -e MAYA_MODEL=Maya-M \
  -e MAYA_API_KEY=changeme \
  ghcr.io/<owner>/project-maya-docker:sm80-latest
```

### Tesla P40 (`sm61`)

Upstream still marks Pascal **experimental**.  A P40 has 24 GB, so several split
the model's layers — this example is a 5×P40 rig (120 GB of VRAM) running every
expert from VRAM:

```sh
mkdir -p /path/on/nvme/models          # ~97-156 GB, on a fast NVMe SSD
docker run -d --name maya-p40 \
  --gpus all \
  --ipc host \
  --cap-add IPC_LOCK \
  --ulimit memlock=-1 \
  -p 8081:8080 \
  -v /path/on/nvme/models:/models \
  -v /path/to/configs:/data \
  -e MAYA_MODEL=Maya-M-Derisked \
  -e MAYA_API_KEY=changeme \
  ghcr.io/<owner>/project-maya-docker:sm61-latest --gpus 0,1,2,3,4
```

The engine drafts with the model's MTP block on Pascal (the DFlash2 drafter
needs RTX 30 / A100 and newer).  Maya-M-Derisked is 116 GB, so on 5×24 GB the
experts nearly fill VRAM and leave little room for the KV cache — keep the
context modest, or pick a smaller quant (`-e MAYA_MODEL=Maya-S24`).

Then open the dashboard (`http://127.0.0.1:8080` for the 170HX example above,
`:8081` for the P40 one) or use the OpenAI-compatible API at `/v1` and the
Anthropic-compatible one at `/v1/messages`.

### First start

The entrypoint runs Maya's non-interactive setup with
`--download-model --models-dir /models`: the quant you chose (default:
**Maya-M-Derisked**; the image sets `MAYA_MODEL`, override it with
`-e MAYA_MODEL=...`) is downloaded **resumably** into `/models` (96.5–156 GB,
plus the vision encoder's 1.1 GB on the `sm80` image), the pack is built next to
it, and the config is kept on
`/data` — a recreated (or updated) container re-uses both.  A stopped download
continues on the next start.  On the `sm80` image the DFlash2 drafter (0.7 GB) is
fetched the same way (see [DFlash2](#dflash2-optional-drafting)).

Note from upstream: while Maya runs, the engine pins almost all free RAM for
its expert tier — other programs get little.  `--ipc host`,
`--cap-add IPC_LOCK` and `--ulimit memlock=-1` let it do so inside the
container (mlock of the ~tens of GB expert pool is required for the RAM tier).

### Text only on the `sm61` image

The Pascal image is built **without** the vision encoder: images need the
encoder's 1.1 GB files and its VRAM beside the expert pool, which a 24 GB P40
rig has little of.  The image's first start therefore passes upstream's
`--no-vision` (nothing is downloaded or compiled), and a `/data` volume from an
image that had the encoder is made text only at start — the config's `vision`
entry is dropped with a note (its files stay in `/models`, unused).  The server
reports text only and refuses image requests.  The gate is the architecture
(`tools/docker_build.py` builds the encoder for sm_70+), so images are on the
`sm80` image and on custom builds for sm_70 and newer.

### Environment

| Variable | Default | Meaning |
|---|---|---|
| `MAYA_MODEL` | `Maya-M-Derisked` | quant for the first download: `Maya-S-v2-IQ2_XXS`, `Maya-S24`, `Maya-M`, `Maya-M-Derisked`, `Maya-L` — set empty for the setup's own recommendation |
| `MAYA_DRAFT` | `dflash2` on `sm80`, `mtp` on `sm61` | how the model drafts: `dflash2` (a DFlash2 drafter, **sm_80+ only**) or `mtp` (the model's own block) |
| `MAYA_NO_SPEED_CALIBRATION` | unset (calibration runs) | with `dflash2`: skip the speed calibration that measures DFlash2 against the MTP block (~5–20 min, loads the model twice) |
| `MAYA_CA_BUNDLE` | unset | path to a PEM bundle of extra CA certificates to trust (internal mirror); also exported as `CURL_CA_BUNDLE`, `SSL_CERT_FILE`, `REQUESTS_CA_BUNDLE` |
| `MAYA_HOST` | `0.0.0.0` | listen address (you are in Docker) |
| `MAYA_PORT` | `8080` | API/dashboard port |
| `MAYA_API_KEY` | unset | require this key (recommended with `0.0.0.0`).  Written into a new config at first start **and** exported as `STRATA_API_KEY` at every start, so it also protects an existing `/data` volume whose config has no `api_key` |
| `MAYA_PREFILL` | unset (upstream's `auto`) | the engine's prompt chunk, exported as the engine's own `STRATA_GLM_PREFILL_CHUNK`: `auto` or a number, e.g. `32768`.  The engine reads that before the config's `--prefill`, so it wins on an existing `/data` volume.  On a multi-GPU split with ≥ 96 GB RAM upstream measures +21-35% prompt reading; below ~64 GB a chunk above 8192 is slower |
| `MAYA_CONTEXT` | unset (the setup's choice, `32768`) | the engine's context length written into the config's `args` (`--max-context`): a number, e.g. `1048576`.  The setup's own `--context` only counts while setting up, so on an existing `/data` volume this is how to change the context |

Any extra arguments are passed to `./maya.sh`, e.g.:

```sh
docker run ... images/... --gpu 0 --parallel 4
```

`MAYA_PREFILL` and `MAYA_CONTEXT` are not given on the `docker run` line.
`MAYA_PREFILL` is exported as the engine's own `STRATA_GLM_PREFILL_CHUNK`, which
the engine reads before the config's `--prefill` (the engine stamps
`STRATA_GLM_PREFILL` itself from the config, so only the `_CHUNK` name works) —
set it with `-e MAYA_PREFILL=32768` and it also applies to an existing `/data`
volume.  `MAYA_CONTEXT` is the context length, i.e. `--max-context` in the
config's `args`, which has no environment equivalent: the entrypoint writes it
into the config at every start.  The setup's own `--context` only counts while
setting up — so on an **existing** `/data` volume `--context 1048576` on the
`docker run` line does nothing, and `-e MAYA_CONTEXT=1048576` is what changes it
(it also wins over a `--context` given on a first start).  On a first start the
setup's log may still print upstream's own "`--prefill 32768`" tip: it is
emitted while the config still says `auto`, and the value applies from the first
engine start.

If the model files come from an **internal mirror** (a private `HF_ENDPOINT`),
trust its CA: mount the certificate(s) read-only into
`/usr/local/share/ca-certificates/` (they are imported at start), or point
`MAYA_CA_BUNDLE` at a PEM bundle. Without it the download stops with
`curl: (60) SSL certificate problem: unable to get local issuer certificate`.

### DFlash2 (optional drafting)

The optional **DFlash2 drafter** — a small block drafter (0.7 GB) that speeds up
answers on **sm_80 and newer** (RTX 30 / A100) — is on by default for the `sm80`
image: on first start it is downloaded into `/models/GLM-5.3-Flash-DFlash2/`
(persistent, resumes if stopped) and the setup's **speed calibration** measures
it against the MTP block (~5–20 min, loads the model twice) and keeps DFlash2
only if it is actually faster on your machine. Turn it off with
`-e MAYA_DRAFT=mtp`; skip the measurement with `-e MAYA_NO_SPEED_CALIBRATION=1`
— useful when the model streams most experts from RAM/SSD and DFlash2 is not
worth the calibration.

On the `sm61` (P40/P100, Pascal) image DFlash2 cannot run (it needs tensor
cores); that image drafts with the MTP block, and asking for `dflash2` there is
ignored with a note.

The drafter is **not baked into the image** (it is CC BY-NC-ND 4.0,
non-commercial use only) — it is downloaded at first start, only when drafting
with DFlash2. Like `MAYA_MODEL`, this is a first-start choice: an existing
`/data` config keeps its drafting mode; change it with a fresh `/data` or a
`--setup` run.

### Rebuild for a different NVIDIA card

```sh
docker build --build-arg CUDA_ARCHITECTURES=89 -t maya-4090 .   # RTX 30/40/50: 86/89/120
docker build --build-arg CUDA_ARCHITECTURES=61 -t maya-p40 .    # Tesla P40 / P100 (Pascal)
```

Arch map: V100/T4 `70;75`, RTX 20 `75`, 30 `86`, 40/Ada `89`, 50/Blackwell
`120`, A100/CMP 170HX `80`, Tesla P40/P100 `61`.  The CUDA base image is pinned
at 12.8.1; its toolchain covers sm_61 (Pascal) and sm_80 - for a Blackwell
build, bump `CUDA_BASE` (e.g. `13.0.1-devel-ubuntu24.04`, which no longer builds
for Pascal or Volta).

Pascal is built differently: upstream's `CMakeLists.txt` refuses any explicit
arch below 7.5 and its setup asks CMake for the build machine's own GPU
(`native`), neither of which works in a GPU-less image build.
`tools/docker_build.py` therefore forces the explicit arch and relaxes that
floor to Maya's own minimum (6.0), so `sm_61` is accepted and upstream's Pascal
path - `-Xptxas -dlcm=ca`, L1-cached global loads the decode needs there - is
picked up automatically.  It also refuses any `MAYA_REF` older than **v1.1.14**,
the release that added Pascal support (PR #214) - an older ref cannot build for
a P40 at all.

### Portable binaries (no `-march=native`)

The images are built on GitHub runners, whose CPUs have AVX-512; the card's own
host often does not (a Xeon E5, a desktop Ryzen).  Upstream's CUDA build leaves
ggml at its default `GGML_NATIVE=ON`, which bakes the build machine's instruction
set into ggml — the engine then dies with `SIGILL` ("Illegal instruction", exit
132) the moment it reaches ggml code, the vision encoder the same.
`tools/docker_build.py` therefore adds upstream's own portable switch to every
cmake configure, `-DSTRATA_PORTABLE=ON -DGGML_NATIVE=OFF`: an AVX2 baseline
everywhere, with ggml still selecting AVX-512 at runtime on CPUs that have it.
(The HIP path sets `STRATA_PORTABLE` itself.)

## Files

- `Dockerfile` — clones upstream at `MAYA_REF` (default latest release handled
  by CI), compiles for `CUDA_ARCHITECTURES`, sets the default `MAYA_MODEL`
- `tools/docker_build.py` — compile-only helper (no GPU needed at build time)
  that builds the engine (with upstream's portable switch: an AVX2 baseline, so
  the image does not carry the build runner's AVX-512) and, on sm_70+, the vision
  encoder, reusing upstream's maya.py machinery; handles Pascal's explicit-arch
  build and enforces the minimum Maya version (v1.1.14)
- `docker-entrypoint.sh` — first-start setup + model download (and the DFlash2
  drafter where it applies), config kept on `/data`
- `.github/workflows/docker-ghcr.yml` — daily tag-resolve + publish of both
  architectures

## Credits

All engine/server/dashboard work is upstream Project Maya (@mw00,
@needmorevram and contributors), MIT.  This repo only packages it.  The
optional DFlash2 drafter is **not baked into the image** (CC BY-NC-ND 4.0,
non-commercial) — it is downloaded at first start, only when drafting with
DFlash2 (the default on the `sm80` image, off on `sm61`).
