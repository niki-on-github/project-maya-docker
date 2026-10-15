Project Maya in Docker — [mw00/project-maya](https://github.com/mw00/project-maya)
(GLM-5.3-Flash, 321 B MoE, MIT) built for **NVIDIA CMP 170HX** (GA100 /
compute capability 8.0) by default.  The engine is compiled in the image for
**one CUDA architecture** (`CUDA_ARCHITECTURES=80`); the model lives outside
the image and downloads on first start.

## CI (GitHub Actions)

A daily cron (this repo, "Build and publish Docker image") resolves upstream's
newest tag, checks whether that tag is published here yet, and otherwise
builds and pushes to GHCR as `ghcr.io/<owner>/<repo>:<upstream-tag>` and
`:latest`.  Manual runs and pushes to `main` always build.

Build inputs (`workflow_dispatch`):

| Input | Default | Meaning |
|---|---|---|
| `maya_ref` | newest upstream tag | upstream ref/tag to build |
| `cuda_arch` | `80` | `CUDA_ARCHITECTURES`, e.g. `89` for an RTX 4090 build |

On a GitHub-hosted runner this drops from ~37 min for all seven upstream
architectures (cold, no cache) to roughly **10–15 min** for the single `sm_80`
arch; with the GHA layer cache warm it is under ~10 min.

## Run

You need the NVIDIA Container Toolkit (`nvidia-container-toolkit`) and a GPU
of compute capability 8.0 (A100, CMP 170HX, GA100 mining cards) for the
default image — or rebuild with your card.

```sh
mkdir -p /path/on/nvme/models          # ~97-156 GB, on a fast NVMe SSD
docker run -d --name maya \
  --gpus all \
  --ipc host \
  --cap-add IPC_LOCK \
  --ulimit memlock=-1 \
  -p 8080:8080 \
  -v /path/on/nvme/models:/models \
  -v /path/to/configs:/data \
  -e MAYA_MODEL=Maya-M \
  -e MAYA_API_KEY=changeme \
  ghcr.io/<owner>/project-maya-docker:latest
```

Then open <http://127.0.0.1:8080> (dashboard) or use the OpenAI-compatible
API at `/v1` and the Anthropic-compatible one at `/v1/messages`.

### First start

The entrypoint runs Maya's non-interactive setup with
`--download-model --models-dir /models`: the quant you chose (default: the
setup's own recommendation for the detected PC) is downloaded **resumably**
into `/models` (96.5–156 GB + 1.1 GB vision encoder), the pack is built next
to it, and the config is kept on `/data` — a recreated (or updated) container
re-uses both.  A stopped download continues on the next start.

Note from upstream: while Maya runs, the engine pins almost all free RAM for
its expert tier — other programs get little.  `--ipc host`,
`--cap-add IPC_LOCK` and `--ulimit memlock=-1` let it do so inside the
container (mlock of the ~tens of GB expert pool is required for the RAM tier).

### Environment

| Variable | Default | Meaning |
|---|---|---|
| `MAYA_MODEL` | setup's recommendation | quant for the first download: `Maya-S-v2-IQ2_XXS`, `Maya-S24`, `Maya-M`, `Maya-M-Derisked`, `Maya-L` |
| `MAYA_HOST` | `0.0.0.0` | listen address (you are in Docker) |
| `MAYA_PORT` | `8080` | API/dashboard port |
| `MAYA_API_KEY` | unset | require this key (recommended with `0.0.0.0`) |

Any extra arguments are passed to `./maya.sh`, e.g.:

```sh
docker run ... images/... --gpu 0 --context 65536 --parallel 4
```

### Rebuild for a different NVIDIA card

```sh
docker build --build-arg CUDA_ARCHITECTURES=89 -t maya-4090 .   # RTX 30/40/50: 86/89/120
```

Arch map: V100/T4 `70;75`, RTX 20 `75`, 30 `86`, 40/Ada `89`, 50/Blackwell
`120`, A100/CMP 170HX `80`.  The CUDA base image is pinned at 12.8.1; its
toolchain covers sm_80 - for a Blackwell build, bump `CUDA_BASE` (e.g.
`13.0.1-devel-ubuntu24.04`).

## Files

- `Dockerfile` — clones upstream at `MAYA_REF` (default latest release handled
  by CI), compiles for `CUDA_ARCHITECTURES`
- `tools/docker_build.py` — compile-only helper (no GPU needed at build time),
  reusing upstream's maya.py machinery
- `docker-entrypoint.sh` — first-start setup + model download, config kept on
  `/data`
- `.github/workflows/docker-ghcr.yml` — daily tag-resolve + publish

## Credits

All engine/server/dashboard work is upstream Project Maya (@mw00,
@needmorevram and contributors), MIT.  This repo only packages it.  The
optional DFlash2 drafter is not included (CC BY-NC-ND, off by default).
