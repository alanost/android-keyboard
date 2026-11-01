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
| `layout` | visible key rectangles | `id kw kh pl pt vw vh late keys[[code,label,x,y,w,h]]` |
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
