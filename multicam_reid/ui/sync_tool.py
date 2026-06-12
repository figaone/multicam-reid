"""
Interactive manual synchronization tool.

Shows all cameras side by side. Scrub each camera independently to a common
visual instant, press A to set the alignment anchor, then mark segment IN/OUT
points and export aligned clips. Multiple segments can be exported separately.

Workflow:
  1. Select a camera (TAB or number keys) and scrub it with the arrows until a
     recognizable event lines up across all cameras.
  2. Press A to set the ANCHOR at the current per-camera positions. This locks
     in the offsets relative to the reference camera (the active one).
  3. Press I and O to mark the segment IN/OUT on the reference timeline.
  4. Press E to export the aligned clips for that segment.
  5. Repeat 3-4 for as many segments as you like.

Exported clips land in <project>/.reid/synced/<segment>/ and can be opened
directly as a new project for tracking + matching.
"""

from __future__ import annotations

import threading
from datetime import datetime

import cv2
import numpy as np
from loguru import logger

from ..core.project import Project
from ..core import sync_io
from ..core.sync_export import export_segment
from .render import draw_text

ACTIVE_COLOR = (0, 255, 0)
INACTIVE_COLOR = (90, 90, 90)
REF_COLOR = (0, 200, 255)
INFO_BAR_HEIGHT = 130
BASE_PANEL_HEIGHT = 430
MAX_CANVAS_WIDTH = 1850
MAX_CANVAS_HEIGHT = 1000

HELP_LINES = [
    "MANUAL SYNC CONTROLS",
    "",
    "TAB / 1..9        Select active camera",
    "R                 Make active camera the REFERENCE",
    "-> / <-           Active cam step +/- 1 frame",
    "] / [             Active cam step +/- 10 frames",
    "} / {             Active cam step +/- 50 frames",
    ". / ,             ALL cams step +/- 1 frame",
    "L / J             ALL cams skip +/- 5 seconds",
    "SPACE             Play / Pause all",
    "+ / -             Playback speed (fast-forward all)",
    "F                 Freeze active cam (others keep playing)",
    "HOME              All cams to frame 0",
    "",
    "A                 Set ANCHOR (lock offsets here)",
    "I / O             Mark segment IN / OUT",
    "                  (O optional - end of video if unset)",
    "C                 Clear current segment marks",
    "E                 Export segment (auto-named, background)",
    "                  Playing to the END auto-exports too",
    "X                 Clear ALL saved segments (confirm)",
    "P                 Print saved segments to console",
    "H                 Toggle this help",
    "Q                 Save and quit",
]


class CamView:
    """Per-camera capture + position state."""

    def __init__(self, cam, video_path):
        self.cam = cam
        self.name = cam.name
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise IOError(f"Could not open video for camera '{cam.name}'")
        self.fps = cam.fps or self.cap.get(cv2.CAP_PROP_FPS) or 10.0
        self.total = cam.frame_count or int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = cam.width
        self.height = cam.height
        self.frame_idx = 0
        self.frozen = False
        self.current = None
        self.read(0)

    def read(self, idx: int) -> None:
        idx = max(0, min(idx, self.total - 1))
        self.frame_idx = idx
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if ret:
            self.current = frame

    def step(self, delta: int) -> None:
        self.read(self.frame_idx + delta)

    @property
    def time_str(self) -> str:
        t = self.frame_idx / self.fps if self.fps else 0
        m, s = divmod(t, 60)
        return f"{int(m):02d}:{s:05.2f}"

    def release(self):
        self.cap.release()


class SyncTool:
    def __init__(self, project: Project):
        self.project = project
        self.cams = [CamView(c, project.video_path(c)) for c in project.cameras]
        self.n = len(self.cams)
        if self.n < 2:
            raise ValueError("Manual sync requires at least 2 cameras.")
        self.names = [c.name for c in self.cams]

        self.sync = sync_io.load_sync(project)
        self.reference = self.sync.get("reference") or self.names[0]
        if self.reference not in self.names:
            self.reference = self.names[0]

        self.active = 0
        self.paused = True
        self.speed = 1
        self.show_help = False
        self.status_msg = "Press H for controls"
        self.seg_in: int | None = None
        self.seg_out: int | None = None
        self.pending_clear_all = False
        self._end_handled = False
        self._export_lock = threading.Lock()
        self._jobs: list[dict] = []

        self._layout()

    # ------------------------------------------------------------------ #
    def _ref_idx(self) -> int:
        return self.names.index(self.reference)

    def _set_status(self, msg: str):
        self.status_msg = msg
        logger.info(f"  {msg}")

    def _layout(self):
        panel_h = BASE_PANEL_HEIGHT

        def widths_for(h):
            return [int(round(h * ((c.width / c.height) if c.height else 16 / 9)))
                    for c in self.cams]

        gap = 6
        widths = widths_for(panel_h)
        total = sum(widths) + gap * (self.n - 1)
        if total > MAX_CANVAS_WIDTH:
            panel_h = int(panel_h * MAX_CANVAS_WIDTH / total)
            widths = widths_for(panel_h)
        self.gap = gap
        self.panel_h = panel_h
        self.panel_w = widths
        self.canvas_w = sum(widths) + gap * (self.n - 1)
        self.canvas_h = panel_h + INFO_BAR_HEIGHT

    # ------------------------------------------------------------------ #
    # Sync actions
    # ------------------------------------------------------------------ #
    def set_anchor(self):
        anchor = {c.name: c.frame_idx for c in self.cams}
        self.sync["anchor"] = anchor
        self.sync["reference"] = self.reference
        self.sync["offsets"] = sync_io.compute_offsets(anchor, self.reference)
        self.save()
        offs = ", ".join(f"{n}:{o:+d}" for n, o in self.sync["offsets"].items())
        self._set_status(f"Anchor set. Offsets [{offs}]")

    def set_reference_to_active(self):
        self.reference = self.names[self.active]
        self.sync["reference"] = self.reference
        if self.sync.get("anchor"):
            self.sync["offsets"] = sync_io.compute_offsets(self.sync["anchor"], self.reference)
        self.save()
        self._set_status(f"Reference camera = {self.reference}")

    def mark_in(self):
        self.seg_in = self.cams[self._ref_idx()].frame_idx
        self._end_handled = False
        self._set_status(f"Segment IN = {self.seg_in} (reference frame)")

    def mark_out(self):
        self.seg_out = self.cams[self._ref_idx()].frame_idx
        self._set_status(f"Segment OUT = {self.seg_out} (reference frame)")

    def clear_segment(self):
        self.seg_in = None
        self.seg_out = None
        self._set_status("Segment marks cleared")

    def clear_all_segments(self):
        """Forget every saved segment for this video set (two-press confirm)."""
        n = len(self.sync.get("segments", []))
        if n == 0:
            self.pending_clear_all = False
            self._set_status("No saved segments to clear")
            return
        if not self.pending_clear_all:
            self.pending_clear_all = True
            self._set_status(f"Press X again to clear ALL {n} saved segments")
            return
        self.sync["segments"] = []
        self.pending_clear_all = False
        self.save()
        self._set_status(f"Cleared all {n} saved segments")

    def export_current_segment(self):
        if self.seg_in is None:
            self._set_status("Mark an IN point (I) before exporting")
            return
        ref = self.cams[self._ref_idx()]
        # If OUT was never marked, run the segment to the end of the video.
        seg_out = self.seg_out if self.seg_out is not None else ref.total - 1
        ref_in, ref_out = sorted((self.seg_in, seg_out))
        if ref_out - ref_in < 2:
            self._set_status("Segment too short to export")
            return

        # Use the CURRENT per-camera positions so every manual adjustment made
        # along the way (lag fixes, freezes, nudges) is reflected in the export.
        offsets = {c.name: c.frame_idx - ref.frame_idx for c in self.cams}
        self.sync["offsets"] = offsets
        self.sync["reference"] = self.reference
        self.save()

        name = self._auto_segment_name()
        out_fps = ref.fps
        job = {"name": name, "status": "running", "thread": None}
        self._jobs.append(job)

        thread = threading.Thread(
            target=self._export_worker,
            args=(self.reference, dict(offsets), ref_in, ref_out, name, out_fps, job),
            daemon=True,
        )
        job["thread"] = thread
        thread.start()
        self._set_status(f"Exporting '{name}' in background - keep working")

        # Ready for the next segment immediately.
        self.seg_in = None
        self.seg_out = None

    def _auto_segment_name(self) -> str:
        """Timestamped, collision-free segment name (no user prompt needed)."""
        base = datetime.now().strftime("seg_%Y%m%d_%H%M%S")
        with self._export_lock:
            taken = {s.get("name") for s in self.sync.get("segments", [])}
        taken |= {j["name"] for j in self._jobs}
        name, i = base, 2
        while name in taken:
            name = f"{base}_{i}"
            i += 1
        return name

    def _export_worker(self, reference, offsets, ref_in, ref_out, name, out_fps, job):
        try:
            out_dir = export_segment(
                self.project, reference, offsets, ref_in, ref_out, name, out_fps
            )
        except (ValueError, IOError) as exc:
            job["status"] = "error"
            self.status_msg = f"Export '{name}' FAILED: {exc}"
            logger.error(f"Export '{name}' failed: {exc}")
            return
        with self._export_lock:
            self.sync["segments"].append({
                "name": name,
                "ref_in": ref_in,
                "ref_out": ref_out,
                "output_fps": out_fps,
                "exported": True,
            })
            self.save()
        job["status"] = "done"
        self.status_msg = f"Exported '{name}'"
        print(f"\n  Exported '{name}' -> {out_dir}\n"
              f"    To track + match it:  python -m multicam_reid match {out_dir}\n")

    def _advance_playback(self):
        """Step all non-frozen cameras one tick; auto-handle end of video."""
        for cam in self.cams:
            if not cam.frozen:
                cam.step(self.speed)
        ref = self.cams[self._ref_idx()]
        if ref.frame_idx >= ref.total - 1:
            # Reached the end of the reference video: stop playing.
            self.paused = True
            if self.seg_in is not None and not self._end_handled:
                # Auto-export the open segment using end-of-video as OUT.
                self._end_handled = True
                self.export_current_segment()
            else:
                self._set_status("Reached end of video")

    def _active_jobs(self) -> int:
        return sum(1 for j in self._jobs if j["status"] == "running")

    def _await_exports(self):
        running = [j for j in self._jobs if j["status"] == "running"]
        if running:
            logger.info(f"  Waiting for {len(running)} export(s) to finish...")
            for j in running:
                t = j.get("thread")
                if t is not None:
                    t.join()

    def print_segments(self):
        segs = self.sync.get("segments", [])
        print(f"\n  --- Saved segments ({len(segs)}) ---")
        for s in segs:
            print(f"    {s['name']}: ref {s['ref_in']}..{s['ref_out']} "
                  f"@ {s.get('output_fps', '?')}fps")
        offs = self.sync.get("offsets", {})
        print(f"  Reference: {self.reference} | Offsets: "
              f"{', '.join(f'{n}:{o:+d}' for n, o in offs.items())}\n")

    def save(self):
        sync_io.save_sync(self.project, self.sync)

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def render(self):
        canvas = np.full((self.canvas_h, self.canvas_w, 3), 30, dtype=np.uint8)
        x = 0
        ref_idx = self._ref_idx()
        for i, cam in enumerate(self.cams):
            pw, ph = self.panel_w[i], self.panel_h
            panel = (cv2.resize(cam.current, (pw, ph)) if cam.current is not None
                     else np.zeros((ph, pw, 3), np.uint8))

            if i == self.active:
                border, bw = ACTIVE_COLOR, 3
            elif i == ref_idx:
                border, bw = REF_COLOR, 2
            else:
                border, bw = INACTIVE_COLOR, 1
            cv2.rectangle(panel, (0, 0), (pw - 1, ph - 1), border, bw)

            tag = " [REF]" if i == ref_idx else ""
            draw_text(panel, f"[{i + 1}] {cam.name}{tag}", (8, 26), scale=0.6,
                      color=ACTIVE_COLOR if i == self.active else (220, 220, 220), weight=2)
            draw_text(panel, f"F:{cam.frame_idx}/{cam.total - 1}", (8, 50),
                      scale=0.5, color=(0, 230, 255), weight=1)
            draw_text(panel, cam.time_str, (8, ph - 12), scale=0.5,
                      color=(220, 220, 220), weight=1)
            if cam.frozen:
                draw_text(panel, "FROZEN", (pw - 110, 26), scale=0.55,
                          color=(0, 80, 255), weight=2)

            canvas[0:ph, x:x + pw] = panel
            x += pw + self.gap

        self._draw_info_bar(canvas)
        if self.show_help:
            self._draw_help(canvas)
        return canvas

    def _draw_info_bar(self, canvas):
        y0 = self.panel_h
        cv2.rectangle(canvas, (0, y0), (self.canvas_w, self.canvas_h), (20, 20, 20), -1)
        cv2.line(canvas, (0, y0), (self.canvas_w, y0), (70, 70, 70), 1)

        offs = self.sync.get("offsets", {})
        anchored = bool(self.sync.get("anchor"))
        offs_str = ", ".join(f"{n}:{o:+d}" for n, o in offs.items())
        line1 = (f"Reference: {self.reference} | "
                 f"Anchor: {'SET' if anchored else 'not set'} | "
                 f"Offsets [{offs_str}]")
        draw_text(canvas, line1, (10, y0 + 26), scale=0.55, color=(255, 255, 255), weight=1)

        seg_in = self.seg_in if self.seg_in is not None else "-"
        seg_out = self.seg_out if self.seg_out is not None else "-"
        n_saved = len(self.sync.get("segments", []))
        active = self._active_jobs()
        exporting = f"  |  Exporting {active}..." if active else ""
        line2 = (f"Segment IN: {seg_in}  OUT: {seg_out}  |  "
                 f"Saved segments: {n_saved}{exporting}  |  {self.status_msg}")
        draw_text(canvas, line2, (10, y0 + 52), scale=0.55, color=(0, 220, 255), weight=1)

        state = "PAUSED" if self.paused else f"PLAYING {self.speed}x"
        draw_text(
            canvas,
            f"{state} | A anchor  I/O in-out  E export  X clear-all  R ref  F freeze  H help  Q quit",
            (10, y0 + 80), scale=0.5, color=(180, 180, 180), weight=1,
        )

        # Progress bar on the reference timeline.
        ref = self.cams[self._ref_idx()]
        progress = ref.frame_idx / max(1, ref.total - 1)
        bar_x, bar_w, bar_y = 10, self.canvas_w - 120, y0 + 100
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + 10), (55, 55, 55), -1)
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + int(bar_w * progress), bar_y + 10),
                      (0, 180, 0), -1)
        # IN/OUT markers on the bar.
        for mark, col in ((self.seg_in, (0, 220, 255)), (self.seg_out, (255, 120, 0))):
            if mark is not None and ref.total > 1:
                mx = bar_x + int(bar_w * (mark / (ref.total - 1)))
                cv2.line(canvas, (mx, bar_y - 3), (mx, bar_y + 13), col, 2)
        draw_text(canvas, f"{100 * progress:.0f}%", (bar_x + bar_w + 10, bar_y + 10),
                  scale=0.45, color=(180, 180, 180), weight=1)

    def _draw_help(self, canvas):
        pad, line_h, box_w = 12, 20, 470
        box_h = pad * 2 + line_h * len(HELP_LINES)
        x0 = (self.canvas_w - box_w) // 2
        y0 = max(10, (self.canvas_h - box_h) // 2)
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.88, canvas, 0.12, 0, canvas)
        cv2.rectangle(canvas, (x0, y0), (x0 + box_w, y0 + box_h), (90, 90, 90), 1)
        for i, line in enumerate(HELP_LINES):
            y = y0 + pad + line_h * (i + 1) - 6
            color = (0, 220, 255) if i == 0 else (235, 235, 235)
            weight = 2 if i == 0 else 1
            draw_text(canvas, line, (x0 + pad, y), scale=0.48, color=color, weight=weight)

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def run(self):
        window = f"multicam_reid sync - {self.project.folder.name}"
        # WINDOW_GUI_NORMAL removes OpenCV's native Qt toolbar (zoom/pan/save),
        # whose blurry tooltips are unreadable; we provide our own H help instead.
        cv2.namedWindow(window, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(window, min(self.canvas_w, MAX_CANVAS_WIDTH),
                         min(self.canvas_h, MAX_CANVAS_HEIGHT))

        print(f"\n  Manual sync - {self.project.folder}")
        print(f"  {'=' * 52}")
        for i, cam in enumerate(self.cams):
            print(f"  [{i + 1}] {cam.name}: {cam.width}x{cam.height} "
                  f"{cam.fps:.2f}fps {cam.total} frames")
        print(f"  Reference: {self.reference}")
        print(f"  Press H in the window for the full controls list.\n")

        while True:
            cv2.imshow(window, self.render())
            wait_ms = 0 if self.paused else max(1, int(1000 / (self.cams[0].fps * self.speed)))
            key = cv2.waitKeyEx(wait_ms)

            if key == -1:
                if not self.paused:
                    self._advance_playback()
                continue

            a = self.active
            if key != ord('x'):
                self.pending_clear_all = False
            if key in (ord('q'), 27):
                self.save()
                self._await_exports()
                logger.info(f"  Saved sync data to {sync_io.sync_path(self.project)}")
                break
            elif key == ord(' '):
                self.paused = not self.paused
            elif key == ord('f'):
                self.cams[a].frozen = not self.cams[a].frozen
                if self.cams[a].frozen:
                    self._set_status(f"{self.cams[a].name} frozen - press SPACE to "
                                     f"play the other cams; nudge this one with arrows")
                else:
                    self._set_status(f"{self.cams[a].name} unfrozen")
            elif key in (ord('+'), ord('=')):
                self.speed = min(self.speed * 2, 16)
            elif key in (ord('-'), ord('_')):
                self.speed = max(self.speed // 2, 1)
            elif key == 9:  # TAB
                self.active = (self.active + 1) % self.n
            elif ord('1') <= key <= ord('9') and (key - ord('1')) < self.n:
                self.active = key - ord('1')
            # Active-cam stepping
            elif key == 65363:  # Right
                self.cams[a].step(1)
            elif key == 65361:  # Left
                self.cams[a].step(-1)
            elif key == ord(']'):
                self.cams[a].step(10)
            elif key == ord('['):
                self.cams[a].step(-10)
            elif key == ord('}'):
                self.cams[a].step(50)
            elif key == ord('{'):
                self.cams[a].step(-50)
            # All-cams stepping
            elif key == ord('.'):
                for cam in self.cams:
                    cam.step(1)
            elif key == ord(','):
                for cam in self.cams:
                    cam.step(-1)
            elif key == ord('l'):
                for cam in self.cams:
                    cam.step(int(5 * cam.fps))
            elif key == ord('j'):
                for cam in self.cams:
                    cam.step(-int(5 * cam.fps))
            elif key == 65360:  # Home
                for cam in self.cams:
                    cam.read(0)
            # Sync actions
            elif key == ord('r'):
                self.set_reference_to_active()
            elif key == ord('a'):
                self.set_anchor()
            elif key == ord('i'):
                self.mark_in()
            elif key == ord('o'):
                self.mark_out()
            elif key == ord('c'):
                self.clear_segment()
            elif key == ord('e'):
                self.export_current_segment()
            elif key == ord('x'):
                self.clear_all_segments()
            elif key == ord('p'):
                self.print_segments()
            elif key == ord('h'):
                self.show_help = not self.show_help
            elif not self.paused:
                self._advance_playback()

        for cam in self.cams:
            cam.release()
        cv2.destroyAllWindows()
