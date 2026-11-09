# Session Replay (modified version of FUTO Keyboard)

This branch is a **modified version** of FUTO Keyboard 0.1.30 for personal, non-commercial
research into personalized typing models. It is not an official FUTO build and is licensed
under the [FUTO Source First License 1.1](LICENSE.md).

It adds an optional recorder (`java/src/org/futo/inputmethod/latin/replay/SessionRecorder.kt`).
It is off by default. Turn it on in *Text prediction → Session-Replay aufzeichnen*.

- Files: one gzipped JSONL per input session in
  `/sdcard/Android/data/org.futo.inputmethod.latin.unstable/files/replay/`.
- Nothing is sent anywhere. The recorder has no network code; files leave via `adb pull`.
- Kept for 90 days, at most 1 GB. *Aufzeichnungen löschen* removes everything immediately.
- Never recorded: password fields, non-text fields (numbers, phone, dates), incognito or
  no-learning fields, code fields, and apps matching the block list (banking, payment,
  authenticator, password managers, lock screen).
- **Limit:** a password typed into a normal text field cannot be detected. Pause recording first.

## Events

Each line is a JSON object with `e` (type) and `t` (`SystemClock.uptimeMillis`, the same clock as
`et`/`dt` in touch events).

| `e` | Meaning | Fields |
|---|---|---|
| `start` | field opened | `v wall build pkg inputType imeOptions field hint label restarting locale w h dpi land` |
| `layout` | visible key rectangles | `id kw kh pl pt vw vh late keys[[code,label,x,y,w,h]]`, plus `shift[dx,dy]` in px while touch correction is on |
| `touch` | raw motion event | `src(main/more) a i et dt ox oy ps[{id,x,y,p,s,maj,min,hist[[t,x,y,p]]}]` |
| `key` | decoded key press and resulting text | `cp kc x y w txt ss se` |
| `sugg` | suggestions shown | `w auto valid l[[word,score,kind,dict]]` |
| `pick` | suggestion tapped | `w to i` |
| `commit` | word finished | `typed to ct sep txt` |
| `revert` | autocorrection undone with backspace | `typed to` |
| `sel` | cursor moved | `os oe ns ne cs ce` |
| `end` | field closed | `reason` |

Files may end without a gzip trailer if the process was killed; readers must tolerate that.

Parse with `tools/replay/replay_extract.py`.

## Touch correction

Optional and off by default (Settings → Text prediction → "Persönliche Trefferkorrektur"). Replay data
shows how far a user's touches land from the key they aim at. With the switch on, every touch on the
main keyboard is moved back by that offset before key detection. The offset comes from
`/sdcard/Android/data/org.futo.inputmethod.latin.unstable/files/calibration.json`, pushed via adb:

```json
{"dx": 0.0, "dy": 0.1}
```

Units are the key pitch (most common key width/height including the gap); positive means the user
lands right of / below the centre; values are clamped to ±0.5. Without the file nothing changes.
The file is re-read whenever an input field starts.

`touch` events keep the raw coordinates. `key` events carry the moved coordinates, and the layout
event of each session records the shift in pixels (`shift`), so raw key positions are `x - shift[0]`,
`y - shift[1]`.
