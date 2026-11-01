#!/usr/bin/env bash
# build_testapp.sh <out.apk>: builds the field test app with plain SDK tools.
set -euo pipefail
OUT="$(realpath -m "$1")"
HERE="$(cd "$(dirname "$0")" && pwd)"
BT="$ANDROID_HOME/build-tools/35.0.0"
JAR="$ANDROID_HOME/platforms/android-35/android.jar"
W="$(mktemp -d)"
"$BT/aapt2" link -o "$W/base.apk" -I "$JAR" --manifest "$HERE/AndroidManifest.xml"
javac -d "$W/classes" -cp "$JAR" --release 11 "$HERE/src/e2e/fields/Main.java"
"$BT/d8" --lib "$JAR" --output "$W" "$W/classes/e2e/fields/"*.class
(cd "$W" && zip -q base.apk classes.dex)
"$BT/zipalign" -f 4 "$W/base.apk" "$W/aligned.apk"
keytool -genkeypair -keystore "$W/k.jks" -storepass e2etest -keypass e2etest -alias e2e \
    -keyalg RSA -keysize 2048 -validity 1000 -dname CN=e2e 2>/dev/null
"$BT/apksigner" sign --ks "$W/k.jks" --ks-pass pass:e2etest --out "$OUT" "$W/aligned.apk"
