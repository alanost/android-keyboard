#!/usr/bin/env python3
"""e2e.py <keyboard.apk> <fields.apk> <artifact-dir>: drives a real keyboard on an emulator and
checks what session replay wrote. Touches are injected, so the whole path from MotionEvent to
pulled file is exercised. Exits non-zero on the first failed check."""
import glob, json, os, re, shutil, subprocess, sys, time, xml.etree.ElementTree as ET, zlib

PKG = "org.futo.inputmethod.latin.unstable"
IME = f"{PKG}/org.futo.inputmethod.latin.LatinIME"
REMOTE = f"/sdcard/Android/data/{PKG}/files/replay"
SETTINGS = f"{PKG}/org.futo.inputmethod.latin.uix.settings.SettingsActivity"
FIELDS = "e2e.fields/.Main"
apk, fields, art = sys.argv[1:4]
os.makedirs(art, exist_ok=True)
passed = []


def adb(*args, check=True):
    r = subprocess.run(["adb", *args], capture_output=True, text=True, timeout=180)
    if check and r.returncode:
        raise RuntimeError(f"adb {args}: {r.stdout}{r.stderr}")
    return r.stdout


def shell(cmd, check=True):
    return adb("shell", cmd, check=check)


def snap(name):
    with open(f"{art}/{name}.png", "wb") as f:
        subprocess.run(["adb", "exec-out", "screencap", "-p"], stdout=f, timeout=60)


def ok(cond, what):
    if not cond:
        snap("failure")
        print(f"FEHLER: {what}")
        sys.exit(1)
    passed.append(what)
    print(f"ok: {what}")


def nodes():
    # uiautomator gives up while the screen is animating, so try a few times
    for _ in range(6):
        shell("rm -f /sdcard/ui.xml; uiautomator dump /sdcard/ui.xml >/dev/null 2>&1", check=False)
        xml = shell("cat /sdcard/ui.xml 2>/dev/null", check=False)
        if xml.strip():
            return list(ET.fromstring(xml).iter("node"))
        time.sleep(2)
    return []


def find_text(pattern):
    for n in nodes():
        if re.search(pattern, n.get("text", ""), re.I):
            a, b, c, d = map(int, re.findall(r"\d+", n.get("bounds")))
            return (a + c) // 2, (b + d) // 2
    return None


def tap_text(pattern, tries=8):
    for _ in range(tries):
        hit = find_text(pattern)
        if hit:
            shell(f"input tap {hit[0]} {hit[1]}")
            time.sleep(1.5)
            return
        shell(f"input swipe {width // 2} {height * 3 // 4} {width // 2} {height // 3} 300")
        time.sleep(1)
    snap("ui")
    ok(False, f"Text auf dem Bildschirm gefunden: {pattern}")


def open_prediction_settings():
    shell(f"am start -n {SETTINGS}")
    time.sleep(4)
    notice = find_text(r"^dismiss$|^schlie.en$|^verwerfen$")
    if notice:
        shell(f"input tap {notice[0]} {notice[1]}")
        time.sleep(1.5)
    for _ in range(4):
        if find_text(r"Aufzeichnungen l"):
            return
        tap_text(r"text prediction|textvorhersage")
        time.sleep(2)
    snap("prediction_settings")
    ok(False, "Vorhersage-Einstellungen geöffnet")


def open_field(kind):
    shell(f"am start -S -n {FIELDS} --es field {kind}")
    for _ in range(15):
        time.sleep(1)
        if "mInputShown=true" in shell("dumpsys input_method"):
            time.sleep(1)
            return
    ok(False, f"Tastatur öffnet sich im Feld {kind}")


def close_field():
    shell("am force-stop e2e.fields")
    time.sleep(2)


def remote_files():
    out = shell(f"ls {REMOTE} 2>/dev/null", check=False)
    return sorted(x for x in out.split() if x.endswith(".jsonl.gz"))


def events(name):
    local = f"{art}/{name}"
    adb("pull", f"{REMOTE}/{name}", local)
    raw = zlib.decompressobj(31).decompress(open(local, "rb").read()).decode()
    return [json.loads(l) for l in raw.splitlines() if l.strip()]


def wait_for_file(known, seconds=15):
    end = time.time() + seconds
    while time.time() < end:
        new = [f for f in remote_files() if f not in known]
        if new:
            return new
        time.sleep(1)
    return []


# A slow emulator trips ANR dialogs in the launcher, which would swallow every tap.
shell("settings put global hide_error_dialogs 1")
shell("am force-stop com.google.android.apps.nexuslauncher", check=False)
time.sleep(20)
adb("install", "-r", "-g", apk)
adb("install", "-r", "-g", fields)
shell(f"ime enable {IME}")
shell(f"ime set {IME}")
adb("logcat", "-b", "crash", "-c", check=False)
width, height = map(int, re.search(r"(\d+)x(\d+)", shell("wm size")).groups())
probe = f"input tap {width // 2} {int(height * 0.8)}"

# Off by default: nothing may be written before the user opts in.
open_field("text")
shell(probe)
time.sleep(1)
snap("keyboard_open")
close_field()
ok(remote_files() == [], "Standardmäßig aus: keine Datei ohne Einschalten")

open_prediction_settings()
snap("settings")
tap_text(r"Session-Replay aufzeichnen")
snap("enabled")
shell("input keyevent KEYCODE_HOME")
time.sleep(1)

# Calibration: one tap anywhere gives us the real layout and the view's position on screen.
open_field("text")
shell(probe)
time.sleep(1)
close_field()
new = wait_for_file([])
ok(len(new) == 1, "Nach dem Einschalten entsteht genau eine Sitzungsdatei")
cal = events(new[0])
kinds = {e["e"] for e in cal}
ok({"start", "layout", "touch", "key", "end"} <= kinds, f"Kalibrierung enthält alle Grundereignisse: {sorted(kinds)}")
ok(cal[0]["pkg"] == "e2e.fields", "start nennt die richtige App")
layout = next(e for e in cal if e["e"] == "layout")
touch = next(e for e in cal if e["e"] == "touch")
ok(len(layout["keys"]) >= 26, f"Layout hat alle Tasten ({len(layout['keys'])})")
rects = {k[0]: k for k in layout["keys"]}
px, py = touch["ox"] + layout["pl"], touch["oy"] + layout["pt"]


def press(ch):
    code, _, x, y, w, h = rects[ord(ch)]
    shell(f"input tap {px + x + w // 2} {py + y + h // 2}")
    time.sleep(0.25)


text = "hallo welt "
known = remote_files()
open_field("text")
for ch in text:
    press(ch)
time.sleep(1)
snap("typed")
close_field()
new = wait_for_file(known)
ok(len(new) == 1, "Tipp-Sitzung wurde gespeichert")
ev = events(new[0])
downs = [e for e in ev if e["e"] == "touch" and e["a"] == 0]
ok(len(downs) == len(text), f"Jeder Tipp ist ein Touch ({len(downs)}/{len(text)})")
keys = [e for e in ev if e["e"] == "key"]
ok("".join(chr(e["cp"]) for e in keys if e["cp"] > 0).lower() == text, "Tastenfolge stimmt")
for e, ch in zip(keys, text):
    _, _, x, y, w, h = rects[ord(ch)]
    if not (x - w <= e["x"] <= x + 2 * w and y - h <= e["y"] <= y + 2 * h):
        ok(False, f"Tipp-Koordinate passt zur Taste {ch!r}: {e['x']},{e['y']}")
passed.append("Tipp-Koordinaten passen zu den Tasten")
ok(any(e["e"] == "sugg" and e["l"] for e in ev), "Vorschläge wurden aufgezeichnet")
commits = [e for e in ev if e["e"] == "commit"]
ok(len(commits) >= 2, f"Wortabschlüsse aufgezeichnet ({len(commits)})")
final = next((e["txt"] for e in reversed(keys) if e.get("txt")), "")
ok("hallo" in final.lower() and "welt" in final.lower(), f"Textstand aufgezeichnet: {final!r}")
times = [e["t"] for e in ev]
ok(times == sorted(times), "Ereignisse sind zeitlich geordnet")
shutil.copy(f"{art}/{new[0]}", f"{art}/sample_session.jsonl.gz")

# Exclusions: tapping in these fields must not create a file at all.
for kind, word in (("password", "geheim"), ("number", "")):
    known = remote_files()
    open_field(kind)
    snap(f"field_{kind}")
    for ch in word:
        press(ch)
    if not word:
        shell(probe)
    time.sleep(1)
    close_field()
    time.sleep(3)
    ok(remote_files() == known, f"Feld '{kind}' wird nicht aufgezeichnet")

r = subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "..", "replay_extract.py"),
                    art, f"{art}/parsed"], capture_output=True, text=True)
ok(r.returncode == 0 and os.path.getsize(f"{art}/parsed/words.jsonl") > 0, f"Auswertung läuft: {r.stdout.strip()[:80]}")

pull_dir = f"{art}/pulled"
r = subprocess.run([os.path.join(os.path.dirname(__file__), "..", "pull.sh"), pull_dir], capture_output=True, text=True)
ok(r.returncode == 0 and glob.glob(f"{pull_dir}/**/*.jsonl.gz", recursive=True), "pull.sh holt die Dateien ohne Root")

# Touch correction: a tap in the upper half of the second row belongs to the first row once the
# calibration says the user lands that far below the key centre.
_, _, wx, wy, ww, wh = rects[ord("w")]
_, _, ax, ay, aw, ah = rects[ord("a")]
pitch = ay - wy
tx, ty = px + wx + int(ww * 0.3), py + wy + wh // 2 + int(pitch * 0.75)


def tap_once():
    known = remote_files()
    open_field("text")
    shell(f"input tap {tx} {ty}")
    time.sleep(1)
    close_field()
    new = wait_for_file(known)
    ev = events(new[0]) if new else []
    return ([chr(e["cp"]).lower() for e in ev if e["e"] == "key" and e["cp"] > 0],
            next((e for e in ev if e["e"] == "layout"), {}))


hit, lay = tap_once()
ok(hit[:1] not in ([], ["w"]) and "shift" not in lay, f"Ohne Trefferkorrektur trifft der Tipp die zweite Reihe ({hit})")
cal = f"{art}/calibration.json"
open(cal, "w").write('{"dx": 0, "dy": 0.45}')
adb("push", cal, REMOTE.rsplit("/", 1)[0] + "/calibration.json")
open_prediction_settings()
tap_text(r"Trefferkorrektur")
snap("correction_enabled")
shell("input keyevent KEYCODE_HOME")
time.sleep(1)
hit, lay = tap_once()
ok(hit[:1] == ["w"] and lay.get("shift", [0, 0])[1] < 0,
   f"Mit Trefferkorrektur wird derselbe Tipp zu 'w' ({hit}, shift={lay.get('shift')})")

open_prediction_settings()
tap_text(r"Aufzeichnungen l")
time.sleep(2)
ok(remote_files() == [], "Löschen entfernt alle Aufzeichnungen")

crashes = adb("logcat", "-b", "crash", "-d", check=False)
open(f"{art}/crash.log", "w").write(crashes)
ok(PKG not in crashes, "Keine Abstürze der Tastatur")
print(f"BESTANDEN: {len(passed)} Prüfungen")
