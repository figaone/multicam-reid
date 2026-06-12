# multicam_reid

A self-contained toolkit for building **cross-camera vehicle re-identification
ground truth**. Point it at a folder of videos (one per camera), let it detect
and track objects, then use the interactive matcher to link the *same physical
object* across cameras. Optionally export a clean ReID image dataset.

Works with **any number of cameras** (2, 3, 4+) and **any video filenames**.

---

## Core concepts — what each command does

The toolkit is a pipeline of five stages. You usually run them in this order,
but each is optional depending on what you already have.

| Command  | What it does | When you need it |
| -------- | ------------ | ---------------- |
| `init`   | Looks inside your folder, finds the video files, reads each one's resolution/fps/length, and records them in a `manifest.json`. | Runs automatically the first time you use any other command. Run it manually only to re-scan a folder. |
| `sync`   | **Time-alignment.** Lets you line the cameras up so the *same instant* is the same frame number in every video, then cut aligned clips. Produces synchronized video files. | Only if your videos are **not already aligned** (different start times, clock drift, manual recording). Skip it if your videos are already synced. |
| `track`  | **Detection + tracking.** Runs YOLO on every camera to find vehicles/people and follow each one across frames, assigning a track ID. Produces per-camera track data. | Always — the matcher needs track boxes to click on. Can be auto-run by `match`. |
| `match`  | **Cross-camera identity linking.** The interactive tool where *you* click the same vehicle in two or more cameras to say "these are the same object". Produces the ground-truth matches. | This is the main human step — the whole point of the tool. |
| `export` | **Dataset generation.** Crops every matched vehicle out of the videos and saves the images grouped by global ID. Produces a ReID training/eval dataset. | When you want image crops for training or evaluating a ReID model. |

### Synchronizing vs. matching — they are different things

These two are easy to confuse, so to be explicit:

- **Synchronizing (`sync`)** is about **time**. It answers: *"Which frame in
  camera B shows the same moment as frame 100 in camera A?"* The output is
  aligned video clips. It does **not** know or care what objects are in the
  scene — it just shifts the videos so their clocks agree.

- **Matching (`match`)** is about **identity**. It answers: *"The red truck
  that camera A calls track 12 is the same red truck camera B calls track 7."*
  It assumes the videos are already time-aligned, and links object IDs across
  cameras. The output is your ground-truth correspondence list.

In short: **sync lines up the clocks, match links the objects.** You sync first
(if needed), then track, then match.

---

## Install

```bash
pip install -r requirements.txt
```

Requires Python 3.10+. Tracking uses [Ultralytics YOLO](https://docs.ultralytics.com/)
with ByteTrack; a GPU is recommended but not required.

## Quick start

Put your camera videos in a single folder (any names, any common format):

```
my_intersection/
├── cam_north.mp4
├── cam_east.mp4
└── cam_west.mp4
```

**Path 1 — videos are already synced (just label them):**

```bash
python -m multicam_reid match my_intersection
```

The first time, `match` notices there are no tracks yet and offers to run
detection + tracking for you. Say yes, wait for it to finish, and the matcher
opens.

**Path 2 — videos are NOT aligned (sync them first):**

```bash
python -m multicam_reid sync   my_intersection   # align + cut aligned segments
python -m multicam_reid match  my_intersection/.reid/synced/seg_20260612_112209
```

**Full pipeline, one step at a time:**

```bash
python -m multicam_reid init   my_intersection   # scan folder (optional, auto-run)
python -m multicam_reid sync   my_intersection   # (optional) align + cut segments
python -m multicam_reid track  my_intersection   # detect + track (cached)
python -m multicam_reid match  my_intersection   # link objects across cameras
python -m multicam_reid export my_intersection   # crop matched objects to a dataset
python -m multicam_reid info   my_intersection   # show status
```

---

## Command reference

Every command takes a `folder` argument: the path to the folder that contains
your camera videos (or, for a synced segment, the path printed by `sync`).

### `init` — scan a folder and create the workspace

```bash
python -m multicam_reid init my_intersection [--force]
```

Discovers the video files in the folder, probes each one (resolution, fps,
frame count), and writes `manifest.json`. This runs automatically the first time
you call any other command, so you rarely need it directly.

| Argument  | Type | Default | Meaning |
| --------- | ---- | ------- | ------- |
| `folder`  | path | —       | Folder containing your camera videos. |
| `--force` | flag | off     | Re-scan and overwrite the existing `manifest.json` (e.g. after you add or rename a video). |

### `sync` — manually align cameras and export segments

```bash
python -m multicam_reid sync my_intersection
```

Opens the interactive alignment window. Scrub each camera independently to a
common visual instant, set an anchor, mark in/out points, and export aligned
clips. Takes no extra arguments — everything is done with the on-screen
controls (press **H** for the list). See
[Manual synchronization](#manual-synchronization-sync) below.

### `track` — run detection + tracking

```bash
python -m multicam_reid track my_intersection \
    --model yolov8x.pt --conf 0.3 --tracker bytetrack.yaml [--force]
```

Runs YOLO detection + a tracker on every camera and caches the result as one
`*.tracks.json` per camera. Skips cameras that already have tracks unless
`--force` is given.

| Argument    | Type  | Default          | Meaning |
| ----------- | ----- | ---------------- | ------- |
| `folder`    | path  | —                | Folder containing your camera videos. |
| `--model`   | str   | `yolov8x.pt`     | YOLO weights file. Any Ultralytics model name works; if it isn't present locally it is downloaded automatically. Smaller = faster, larger = more accurate (`yolov8n/s/m/l/x.pt`). |
| `--conf`    | float | `0.3`            | Detection confidence threshold (0–1). Lower finds more objects but more false positives; higher is stricter. |
| `--tracker` | str   | `bytetrack.yaml` | Tracker config. `bytetrack.yaml` (default) or `botsort.yaml`. |
| `--force`   | flag  | off              | Re-run tracking even if cached tracks already exist (overwrites them). |

### `match` — link objects across cameras (the main step)

```bash
python -m multicam_reid match my_intersection [--frame 0] \
    [--model ... --conf ... --tracker ... --force]
```

Opens the interactive matcher. If tracks are missing it offers to run tracking
first (using the same `--model/--conf/--tracker/--force` options as `track`).
See [Matching workflow](#matching-workflow) below.

| Argument    | Type | Default | Meaning |
| ----------- | ---- | ------- | ------- |
| `folder`    | path | —       | Folder containing your camera videos (or a `.reid/synced/<segment>` path). |
| `--frame`   | int  | `0`     | Frame number to start the matcher at. |
| `--model` / `--conf` / `--tracker` / `--force` | — | — | Same as `track`; only used if tracking has to be run because no tracks are cached yet. |

### `export` — crop matched objects into a ReID dataset

```bash
python -m multicam_reid export my_intersection \
    --samples 5 --min-size 16 --padding 0.0
```

Goes through every confirmed match, treats it as one global ID, and crops that
object out of each camera across several frames. Images are grouped per ID.

| Argument      | Type  | Default | Meaning |
| ------------- | ----- | ------- | ------- |
| `folder`      | path  | —       | Folder containing your camera videos. |
| `--samples`   | int   | `5`     | How many frames to crop **per camera per matched ID**. More = more images but slower. |
| `--min-size`  | int   | `16`    | Skip crops smaller than this many pixels on a side (filters out tiny/distant detections). |
| `--padding`   | float | `0.0`   | Extra context added around each box, as a fraction of box size (`0.1` = +10% on each side). |

### `info` — print project status

```bash
python -m multicam_reid info my_intersection
```

Prints the cameras, their resolution/fps/length, how many tracks each has, and
how many matches exist. Takes only the `folder` argument.

---

## Manual synchronization (`sync`)

If your camera videos are **not already aligned**, the `sync` tool lets you line
them up by hand and cut out one or more aligned segments.

```bash
python -m multicam_reid sync my_intersection
```

How it works:

1. Each camera is shown side by side. Select one (`TAB` or number keys) and
   scrub it with the arrow keys until a recognizable event (a car passing, a
   light change) lines up across all cameras.
2. Press **A** to set the **anchor** — this locks in each camera's frame offset
   relative to the *reference* camera (the active one; press **R** to change which
   camera is the reference).
3. Press **I** to mark the segment **in** point on the reference timeline, scrub
   forward, then **O** to mark the **out** point. (If you never press **O**, the
   segment runs to the **end of the video**.)
4. Press **E** to export. The segment is **auto-named** with a timestamp
   (e.g. `seg_20260612_112209`) and written **in the background** — you can keep
   scrubbing and mark/export the next segment immediately while it saves.
   Letting playback run to the **end of the video** also auto-exports the open
   segment (no key press needed).
5. Repeat 3–4 to cut as many segments as you like — each is saved separately.

**Fine-tuning alignment while playing.** Press **F** to *freeze* the active
camera: it stays on its current frame while the **other** cameras keep playing
when you press **SPACE**. This lets you correct a small lag on one camera, then
press **SPACE** to continue, freeze/nudge again later, and so on. The export
always uses the camera positions **as they are when you press `E`**, so every
adjustment you made along the way is reflected in the final clip.

> **One offset per camera.** Each camera is exported with a single constant
> offset for the whole segment. That perfectly captures a fixed lag correction.
> If a camera *drifts continuously* within a segment, split it into shorter
> segments instead.
>
> **Equal length.** All exported clips for a segment have exactly the same
> number of frames. If a camera's offset would run past the start or end of its
> source video, the segment is automatically trimmed to the range every camera
> can cover — so there are no duplicated or frozen padding frames.

Before the window opens, `sync` prints each camera's frame rate and warns if
they differ — a constant offset can drift over long segments when the cameras
run at different fps. Re-encode to a common rate first if needed, e.g.
`ffmpeg -i input.mp4 -r 10 -c:v libx264 output.mp4`.

Exported clips land in `.reid/synced/<segment>/` and are themselves valid
projects, so you can run tracking and matching on a single segment:

```bash
python -m multicam_reid match my_intersection/.reid/synced/seg_20260612_112209
```

> **What is the "reference" camera?** Offsets are measured relative to one
> camera you pick (its offset is always 0). For example, if you anchor with
> camera B 12 frames behind camera A and pick A as reference, B's offset is −12,
> meaning "to see B's view of A-frame 100, read B-frame 88". The reference also
> defines the timeline your IN/OUT marks refer to.


### Manual sync controls

Press **H** in the sync window for this list any time.

| Key            | Action                                       |
| -------------- | -------------------------------------------- |
| `TAB` / `1..9` | Select active camera                         |
| `R`            | Make the active camera the **reference**     |
| `→` / `←`      | Active cam step ±1 frame                      |
| `]` / `[`      | Active cam step ±10 frames                   |
| `}` / `{`      | Active cam step ±50 frames                   |
| `.` / `,`      | Step **all** cameras ±1 frame                |
| `L` / `J`      | Skip **all** cameras ±5 seconds              |
| `SPACE`        | Play / Pause all                             |
| `F`            | Freeze active cam (others keep playing on `SPACE`) |
| `+` / `-`      | Playback speed (fast-forward all when playing) |
| `HOME`         | All cameras to frame 0                        |
| `A`            | Set the alignment **anchor**                 |
| `I` / `O`      | Mark segment **in** / **out** (`O` optional — end of video if unset) |
| `C`            | Clear the current segment marks              |
| `E`            | Export segment (auto-named, saves in background; end of video auto-exports) |
| `X`            | Clear **all** saved segments (press twice to confirm) |
| `P`            | Print saved segments to the console          |
| `H`            | Toggle help overlay                          |
| `Q`            | Save and quit (waits for exports to finish)  |

## How data is stored

The toolkit never scatters files around. Everything it derives from your videos
(detections, tracks, matches, exports) lives in a single hidden `.reid/`
workspace **inside your own folder**, so re-opening the folder restores all
state automatically and you can copy/share the whole folder as one unit.

```
my_intersection/                   <- the folder YOU point the tool at
├── cam_north.mp4                  <- your source videos (any names/format)
├── cam_east.mp4
├── cam_west.mp4
└── .reid/                         <- created automatically by the tool
    ├── manifest.json              # one entry per camera:
    │                              #   name, video filename, width, height,
    │                              #   fps, frame_count, frame_offset
    ├── tracks/                    # detection + tracking output (cached)
    │   ├── cam_north.tracks.json  #   one file per camera, named <camera>.tracks.json
    │   ├── cam_east.tracks.json   #   each = {track_id: {frames, boxes, classes,
    │   └── cam_west.tracks.json   #            confs, class_name}}
    ├── sync.json                  # manual alignment: reference, offsets, segments
    ├── synced/                    # exported aligned segments (after `sync`)
    │   └── segment_01/            #   each is itself a valid project folder
    │       ├── cam_north_synced.mp4
    │       ├── cam_east_synced.mp4
    │       └── cam_west_synced.mp4
    ├── matches.json               # your cross-camera ground-truth matches
    └── export/                    # ReID crop dataset (only after `export`)
        ├── id_0001/               #   one folder per matched global ID
        │   ├── cam_north_f000123.jpg
        │   └── cam_east_f000130.jpg
        ├── id_0002/
        └── export_summary.json
```

### Where each step writes

| Step                              | Reads                          | Writes                          |
| --------------------------------- | ------------------------------ | ------------------------------- |
| `init`                            | your video files               | `.reid/manifest.json`           |
| `sync`                            | videos + manifest              | `.reid/sync.json`, `.reid/synced/<segment>/*.mp4` |
| `track`                           | videos + manifest              | `.reid/tracks/<camera>.tracks.json` |
| `match`                           | videos + tracks                | `.reid/matches.json` (auto-saves on every confirm) |
| `export`                          | videos + tracks + matches      | `.reid/export/id_XXXX/*.jpg`    |

**Camera naming:** each camera's stable `name` is derived from its video
filename (e.g. `cam_north.mp4` -> `cam_north`) and stored in the manifest. That
name is used consistently for its tracks file, its key in `matches.json`, and
its crop filenames, so everything stays linked even if you add cameras later.

**Important:** `matches.json` is your annotation work and is saved automatically
every time you confirm a match. Don't delete the `.reid/` folder unless you mean
to discard all tracks and matches for that project.


## Matcher controls

Press **H** in the matcher window to see this list any time.

| Key                | Action                                   |
| ------------------ | ---------------------------------------- |
| Left click box     | Select / link a track                    |
| Click matched box  | Edit that match group                    |
| `ENTER`            | Confirm current match                    |
| `BACKSPACE`        | Clear current selection                  |
| `N`                | Jump to next unmatched track             |
| `D`                | Delete match — the one being edited, else the last (asks to confirm) |
| `X`                | Clear **all** matches (asks to confirm)  |
| `Ctrl+Z` / `Ctrl+Y`| Undo / Redo                              |
| `SPACE`            | Play / Pause                             |
| `.` / `,`          | Step ±1 frame                            |
| `→` / `←`          | Step ±10 frames                          |
| `W` / `S`          | Skip ±5 seconds                          |
| `TAB`              | Print the match list to the console      |
| `H`                | Toggle help overlay                      |
| `Q`                | Save and quit                            |

### Matching workflow

1. Click a vehicle's box in one camera — it highlights in the next match color.
2. Click the *same* vehicle in another camera to link them.
3. Press `ENTER` to confirm. The group gets a unique persistent color.
4. To add a camera to an existing match later, just click any of its boxes to
   enter **edit mode**, add the missing camera, and confirm.
5. Use `N` to hop to the next vehicle that still needs matching.

**Box colors in the matcher:**

- **Light gray** — an unmatched track (detected, but not yet linked to anything).
- **A solid bright color** — a confirmed match. Every match group has its own
  unique, persistent color, shown identically across all cameras, so you can
  see at a glance which boxes belong to the same object.
- **Thick highlighted box** — your current selection (the link you're building
  right now), shown in the color the next confirmed match will receive.

> Tracking flags (`--model`, `--conf`, `--tracker`, `--force`) are documented in
> the [Command reference](#command-reference). `match` only uses them when it
> has to run tracking because no tracks are cached yet.

## File formats

**Tracks** (`<cam>.tracks.json`) — one file per camera, written by `track`:

```json
{
  "12": {
    "frames": [100, 101, 102],
    "boxes": [[x1, y1, x2, y2], ...],
    "classes": [2, 2, 2],
    "confs": [0.91, 0.88, 0.90],
    "class_name": "car"
  }
}
```

The top-level key (`"12"`) is the track ID. `frames` and `boxes` are parallel
lists: `boxes[i]` is the pixel box at frame `frames[i]`, in original video
resolution.

**Matches** (`matches.json`) — written by `match`:

```json
{
  "version": 1,
  "matches": [
    { "frame": 250, "tracks": { "cam_north": 12, "cam_east": 7, "cam_west": null } }
  ]
}
```

Each entry is one global object. `frame` is the reference frame where you made
the link. `tracks` maps each camera name to the track ID it gave that object,
or `null` if the object was not visible / not linked in that camera.

**Sync** (`sync.json`) — written by `sync`:

```json
{
  "version": 1,
  "reference": "cam_north",
  "anchor": { "cam_north": 420, "cam_east": 408, "cam_west": 425 },
  "offsets": { "cam_north": 0, "cam_east": -12, "cam_west": 5 },
  "segments": [
    { "name": "segment_01", "ref_in": 100, "ref_out": 700,
      "output_fps": 10.0, "exported": true }
  ]
}
```

`offsets[cam]` is the frame shift relative to `reference`; the synced frame for
camera `cam` at reference frame `r` is `r + offsets[cam]`. Each `segment` records
the in/out window (on the reference timeline) it was cut from.

