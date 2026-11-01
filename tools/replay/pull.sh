#!/usr/bin/env bash
# pull.sh <target-dir> [adb-serial]: fetch replay sessions from the phone.
set -eu
DST="$1"; SERIAL="${2:-}"
SRC=/sdcard/Android/data/org.futo.inputmethod.latin.unstable/files/replay
mkdir -p "$DST"; chmod 700 "$DST"
adb ${SERIAL:+-s "$SERIAL"} pull "$SRC/." "$DST/"
