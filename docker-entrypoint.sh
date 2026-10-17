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
#   MAYA_HOST      where the API listens        (default 0.0.0.0 - you are in Docker)
#   MAYA_PORT      the API's port               (default 8080)
#   MAYA_API_KEY   require this key (recommended when listening on 0.0.0.0)
# Any command-line arguments are passed to ./maya.sh (e.g. --gpu 0 --context 65536).

ROOT=/opt/project-maya
CONFDIR=/data/maya/configs
cd "$ROOT" || exit 1

mkdir -p "$CONFDIR" /models

# Maya writes its configs into its own folder (which is part of the image):
# link the ones on the /data volume in, so reads AND writes go to the volume.
for f in "$CONFDIR"/maya-*.json; do
    [ -e "$f" ] || break
    ln -sf "$f" "$ROOT/$(basename "$f")"
done

have_config() { ls "$CONFDIR"/maya-*.json >/dev/null 2>&1; }

if ! have_config; then
    echo "== Project Maya first start: setup, model download (resumes when stopped) =="
    args="--setup --yes --plain --no-start --download-model --models-dir /models"
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

exec ./maya.sh "$@"
