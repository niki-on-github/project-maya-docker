#!/bin/sh
# Project Maya in Docker - entrypoint.
#
# First container start: runs Maya's non-interactive setup, which downloads the
# model (resumable, GitHub LFS-backed) into /models, builds the pack and writes
# the config - which is then kept on the /data volume, so a recreated container
# does not re-download anything.  Later starts simply start Maya.
#
# Environment:
#   MAYA_MODEL     which quant to download on first start: Maya-S-v2-IQ2_XXS,
#                  Maya-S24, Maya-M, Maya-M-Derisked, Maya-L (the image sets
#                  Maya-M-Derisked; override with -e MAYA_MODEL=..., or an empty
#                  value for the setup's own recommendation)
#   MAYA_DRAFT     how the model drafts on first start: dflash2 = a DFlash2
#                  drafter (needs sm_80+ - RTX 30 / A100), mtp = the model's own
#                  MTP block.  Default: dflash2 on an sm_80+ image, mtp otherwise
#   MAYA_NO_SPEED_CALIBRATION
#                  set (any value) to keep DFlash2 without measuring it against
#                  the MTP block (that calibration loads the model twice)
#   MAYA_CA_BUNDLE a PEM bundle of extra CA certificates to trust (for an internal
#                  Hugging Face mirror, say); certificates mounted into
#                  /usr/local/share/ca-certificates/*.crt are imported as well
#   MAYA_HOST      where the API listens        (default 0.0.0.0 - you are in Docker)
#   MAYA_PORT      the API's port               (default 8080)
#   MAYA_API_KEY   require this key: written into a new config at first start,
#                  and exported as STRATA_API_KEY at every start (so an existing
#                  config written without one is protected too)
#   MAYA_PREFILL   the engine's prompt chunk, exported as STRATA_GLM_PREFILL_CHUNK:
#                  auto or a number, e.g. 32768 (unset: leave upstream's "auto")
#   MAYA_CONTEXT   the engine's context length written into the config's args: a
#                  number, e.g. 1048576 (unset: leave the setup's choice).  The
#                  setup option behind it, --context, only counts while setting up,
#                  so on an existing /data volume this is how to change it
# Any command-line arguments are passed to ./maya.sh (e.g. --gpu 0 --parallel 4).

ROOT=/opt/project-maya
CONFDIR=/data/maya/configs
cd "$ROOT" || exit 1

mkdir -p "$CONFDIR" /models

# Trust a private CA (an internal Hugging Face mirror, say).  Certificates mounted
# into /usr/local/share/ca-certificates/*.crt are imported into the system store,
# and MAYA_CA_BUNDLE points curl, OpenSSL and Python at a PEM bundle directly -
# without this, a mirror whose certificate is not publicly trusted fails with
# "curl: (60) SSL certificate problem: unable to get local issuer certificate".
if ls /usr/local/share/ca-certificates/*.crt >/dev/null 2>&1; then
    update-ca-certificates >/dev/null 2>&1 || echo "warning: update-ca-certificates failed" >&2
fi
if [ -n "${MAYA_CA_BUNDLE:-}" ]; then
    export CURL_CA_BUNDLE="$MAYA_CA_BUNDLE" \
           SSL_CERT_FILE="$MAYA_CA_BUNDLE" \
           REQUESTS_CA_BUNDLE="$MAYA_CA_BUNDLE"
fi

# Maya writes its configs into its own folder (which is part of the image):
# link the ones on the /data volume in, so reads AND writes go to the volume.
for f in "$CONFDIR"/maya-*.json; do
    [ -e "$f" ] || break
    ln -sf "$f" "$ROOT/$(basename "$f")"
done

have_config() { ls "$CONFDIR"/maya-*.json >/dev/null 2>&1; }

# An engine flag that maya.py only reads while setting up (so on an existing /data
# volume the config's value would win): keep the config's args in sync with the
# environment on every start.  $1 = the flag, $2 = its value, $3 = the log label.
# A flag that is not there yet is appended; an unset value leaves the config alone.
apply_engine_arg() {
    [ -n "${2:-}" ] || return 0
    MAYA_ARG_FLAG="$1" MAYA_ARG_VALUE="$2" MAYA_ARG_LABEL="$3" CONFDIR="$CONFDIR" python3 - <<'PY'
import glob, json, os, sys
flag, value, label = (os.environ[k] for k in ("MAYA_ARG_FLAG", "MAYA_ARG_VALUE", "MAYA_ARG_LABEL"))
for p in sorted(glob.glob(os.path.join(os.environ["CONFDIR"], "maya-*.json"))):
    try:
        cfg = json.load(open(p))
    except (OSError, ValueError) as e:      # never break the start over this
        print(f"warning: {p}: {e}", file=sys.stderr)
        continue
    args = cfg.get("args") or []
    if flag in args[:-1]:
        if args[args.index(flag) + 1] == value:
            continue                        # already what the environment says
        args[args.index(flag) + 1] = value
    else:
        args += [flag, value]
    cfg["args"] = args
    tmp = f"{p}.tmp"                        # write beside it and rename: a kill in the
    with open(tmp, "w") as f:                # middle must not truncate the config
        json.dump(cfg, f, indent=1)          # indent=1: upstream's own style
    os.replace(tmp, p)
    print(f"{label}: {p}: {flag} {value}")
PY
}

if ! have_config; then
    echo "== Project Maya first start: setup, model download (resumes when stopped) =="
    # How the model drafts tokens ahead.  DFlash2 needs compute capability 8.0+
    # (RTX 30 / A100) and is the default there; elsewhere the model's own MTP
    # block.  CUDA_ARCHITECTURES is the arch this image was built for (Dockerfile);
    # the lowest of the list decides, as upstream's own check does.
    arch=$(printf '%s\n' "${CUDA_ARCHITECTURES:-}" | tr ',;' '\n\n' \
           | grep -oE '^[0-9]+' | sort -n | head -1)
    draft=${MAYA_DRAFT:-}
    if [ -z "$draft" ]; then
        if [ -n "$arch" ] && [ "$arch" -ge 80 ]; then draft=dflash2; else draft=mtp; fi
    fi
    if [ "$draft" = dflash2 ] && { [ -z "$arch" ] || [ "$arch" -lt 80 ]; }; then
        echo "note: DFlash2 needs compute capability 8.0+ (RTX 30 / A100); this image is sm_${arch:-unknown} - using the MTP block" >&2
        draft=mtp
    fi
    args="--setup --yes --plain --no-start --download-model --models-dir /models"
    args="$args --draft $draft"
    [ -n "$MAYA_NO_SPEED_CALIBRATION" ] && args="$args --no-speed-calibration"
    args="$args --host ${MAYA_HOST:-0.0.0.0}"
    [ -n "$MAYA_PORT" ] && args="$args --port $MAYA_PORT"
    [ -n "$MAYA_MODEL" ] && args="$args --model $MAYA_MODEL"
    [ -n "$MAYA_API_KEY" ] && args="$args --api-key $MAYA_API_KEY"
    # shellcheck disable=SC2086
    ./maya.sh $args "$@" || exit 1
    if ! ls "$ROOT"/maya-*.json >/dev/null 2>&1; then
        echo "setup did not install a model/config - see the log above" >&2
        exit 1
    fi
    echo "== Keeping the config(s) on the /data volume =="
    for f in "$ROOT"/maya-*.json; do
        [ -f "$f" ] || break
        mv "$f" "$CONFDIR/$(basename "$f")"
        ln -sf "$CONFDIR/$(basename "$f")" "$f"
    done
fi

# Both paths have converged here (a first start moved its configs onto /data), so
# what follows covers a new install and an existing /data volume alike.
# MAYA_CONTEXT: the context length, i.e. --max-context in the config's args.  The
# setup's own menu stops at 128K, and its --context only counts while setting up -
# written after it, a value here wins either way.
case "${MAYA_CONTEXT:-}" in
    "") ;;                            # unset: leave the config as it is
    *[!0-9]*) echo "warning: ignoring MAYA_CONTEXT=$MAYA_CONTEXT (use a number, e.g. 1048576)" >&2
              MAYA_CONTEXT= ;;
esac
apply_engine_arg --max-context "${MAYA_CONTEXT:-}" MAYA_CONTEXT

# MAYA_PREFILL: the engine's prompt chunk, exported as the engine's own
# STRATA_GLM_PREFILL_CHUNK - the one the engine reads before the config's
# "--prefill", so it wins on an existing /data volume without touching it.  (The
# engine stamps STRATA_GLM_PREFILL itself from the config's --prefill, so only the
# _CHUNK name works.)  Upstream writes "--prefill auto"; on a multi-GPU split with
# plenty of RAM an explicit 32768 reads prompts 21-35% faster (a chunk above 8192
# is slower below ~64 GB).
case "${MAYA_PREFILL:-}" in
    "" | auto) ;;                     # auto is what upstream writes anyway
    *[!0-9]*) echo "warning: ignoring MAYA_PREFILL=$MAYA_PREFILL (use 'auto' or a number)" >&2 ;;
    *) export STRATA_GLM_PREFILL_CHUNK="$MAYA_PREFILL" ;;
esac

# The server gates /v1 on STRATA_API_KEY (serve/server.py: --api-key defaults to
# it), and maya.py passes the whole environment through.  Exporting MAYA_API_KEY
# as STRATA_API_KEY also protects an existing /data volume whose config was
# written without an "api_key" (only a fresh setup writes one).  Unset: no
# change; a STRATA_API_KEY given directly is left intact.
export STRATA_API_KEY="${MAYA_API_KEY:-${STRATA_API_KEY:-}}"

exec ./maya.sh "$@"
