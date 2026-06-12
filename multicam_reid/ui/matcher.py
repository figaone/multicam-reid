"""
Interactive cross-camera track matcher.

Generalized to any number of cameras. Reads cached tracks from a Project's
workspace and lets the user link the same physical object across cameras to
build cross-camera ground-truth matches.

Features:
  - Unique, persistent color per match group (palette cycled by match index)
  - Edit mode: click an already-matched track to add/replace cameras in it
  - Multi-level undo/redo for add / delete / edit actions
  - Click-by-click selection undo
  - Next-unmatched navigation
  - Mouse hover highlight
  - On-screen help overlay (toggle with H)
  - Confirmation prompts for destructive actions
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from ..core.project import Project
from ..core import tracks_io
from .render import draw_text, measure_text

# Per-camera label colors (cycled if more cameras than entries).
CAM_COLORS = [
    (0, 255, 0),      # green
    (255, 165, 0),    # orange
    (255, 0, 255),    # magenta
    (0, 255, 255),    # yellow
    (255, 128, 0),    # sky blue
    (128, 255, 0),    # chartreuse
]

UNMATCHED_COLOR = (210, 210, 210)  # light gray — clearly visible against video
HOVER_COLOR = (255, 255, 255)      # white outline on hover

# Distinct colors for matched groups (cycled by match index).
MATCH_PALETTE = [
    (0, 0, 255), (255, 0, 0), (0, 200, 0), (255, 0, 255), (0, 165, 255),
    (255, 255, 0), (128, 0, 255), (0, 255, 128), (255, 128, 0), (0, 128, 255),
    (180, 105, 255), (0, 255, 255), (255, 100, 100), (100, 255, 100),
    (100, 100, 255), (200, 200, 0), (200, 0, 200), (0, 200, 200),
    (128, 255, 0), (255, 0, 128),
]

MAX_CANVAS_WIDTH = 1850
MAX_CANVAS_HEIGHT = 1000
BASE_PANEL_HEIGHT = 460
INFO_BAR_HEIGHT = 100

HELP_LINES = [
    "CONTROLS",
    "",
    "Left click box      Select / link a track",

    "Click matched box   Edit that match group",
    "ENTER               Confirm current match",
    "BACKSPACE           Clear current selection",
    "N                   Jump to next unmatched track",
    "D                   Delete match (edited one, else last)",
    "X                   Clear ALL matches (confirm)",
    "Ctrl+Z / Ctrl+Y     Undo / Redo",
    "SPACE               Play / Pause",
    ". / ,               Step +/- 1 frame",
    "-> / <-             Step +/- 10 frames",
    "W / S               Skip +/- 5 seconds",
    "TAB                 Print match list to console",
    "H                   Toggle this help",
    "Q                   Save and quit",
]


class Matcher:
    def __init__(self, project: Project, start_frame: int = 0):
        self.project = project
        self.cameras = project.cameras
        self.cam_names = project.camera_names
        self.n_cams = len(self.cameras)
        if self.n_cams < 2:
            raise ValueError("Matcher requires at least 2 cameras.")

        # Open video captures and load tracks.
        self.caps = []
        self.tracks: dict[str, dict] = {}
        for cam in self.cameras:
            cap = cv2.VideoCapture(str(project.video_path(cam)))
            if not cap.isOpened():
                raise IOError(f"Could not open video for camera '{cam.name}'")
            self.caps.append(cap)
            self.tracks[cam.name] = tracks_io.load_tracks(project.tracks_path(cam))
            logger.info(f"  Loaded {len(self.tracks[cam.name])} tracks for '{cam.name}'")

        self.fps = self.cameras[0].fps or self.caps[0].get(cv2.CAP_PROP_FPS) or 10.0
        self.total_frames = min(c.frame_count for c in self.cameras)
        self.frame_idx = max(0, min(start_frame, self.total_frames - 1))

        # Matches: list of {"frame": int, "tracks": {cam_name: tid_or_None}}
        self.matches = tracks_io.load_matches(project.matches_path)
        logger.info(f"  Loaded {len(self.matches)} existing matches")

        # Selection / edit state.
        self.selected: dict[str, int] = {}          # cam_name -> track_id
        self.selection_history: list[dict] = []      # stack for click-undo
        self.editing_match_idx: int | None = None
        self.matched_tracks = self._build_matched_set()
        self.match_color_map = self._build_color_map()
        self.current_select_color = self._next_color()

        # Undo / redo of confirmed actions.
        self.undo_stack: list[tuple] = []
        self.redo_stack: list[tuple] = []

        # UI state.
        self.paused = True
        self.show_help = False
        self.hover: tuple[int, int] | None = None     # (cam_idx, track_id)
        self.pending_confirm: tuple[str, str] | None = None  # (action, message)
        self._pending_delete_idx: int | None = None
        self.status_msg = ""

        self._layout()

    # ------------------------------------------------------------------ #
    # Layout
    # ------------------------------------------------------------------ #
    def _layout(self) -> None:
        """Compute a single-row panel layout that fits the screen budget."""
        panel_h = BASE_PANEL_HEIGHT
        # Each panel width preserves the camera's aspect ratio at panel_h.
        def widths_for(h: int) -> list[int]:
            ws = []
            for cam in self.cameras:
                ar = (cam.width / cam.height) if cam.height else (16 / 9)
                ws.append(int(round(h * ar)))
            return ws

        gap = 6
        widths = widths_for(panel_h)
        total = sum(widths) + gap * (self.n_cams - 1)
        if total > MAX_CANVAS_WIDTH:
            scale = MAX_CANVAS_WIDTH / total
            panel_h = int(panel_h * scale)
            widths = widths_for(panel_h)

        self.gap = gap
        self.panel_h = panel_h
        self.panel_w = widths  # per-camera panel widths
        # Per-camera scale = displayed / original.
        self.scales = [
            (widths[i] / self.cameras[i].width) if self.cameras[i].width else 1.0
            for i in range(self.n_cams)
        ]
        self.canvas_w = sum(widths) + gap * (self.n_cams - 1)
        self.canvas_h = panel_h + INFO_BAR_HEIGHT

    # ------------------------------------------------------------------ #
    # Color / match helpers
    # ------------------------------------------------------------------ #
    def _match_tracks(self, m: dict) -> dict:
        """Return the cam_name -> track_id mapping for a match dict."""
        return m.get("tracks", {})

    def _build_matched_set(self) -> set:
        matched = set()
        for m in self.matches:
            for cam_name, tid in self._match_tracks(m).items():
                if tid is not None:
                    matched.add((cam_name, tid))
        return matched

    def _build_color_map(self) -> dict:
        color_map = {}
        for i, m in enumerate(self.matches):
            color = MATCH_PALETTE[i % len(MATCH_PALETTE)]
            for cam_name, tid in self._match_tracks(m).items():
                if tid is not None:
                    color_map[(cam_name, tid)] = color
        return color_map

    def _next_color(self):
        return MATCH_PALETTE[len(self.matches) % len(MATCH_PALETTE)]

    def _find_match_for_track(self, cam_name: str, track_id: int):
        for i, m in enumerate(self.matches):
            if self._match_tracks(m).get(cam_name) == track_id:
                return i
        return None

    def _load_match_into_selection(self, match_idx: int) -> None:
        m = self.matches[match_idx]
        self.selected = {
            cam: tid for cam, tid in self._match_tracks(m).items() if tid is not None
        }
        self.editing_match_idx = match_idx
        self.current_select_color = MATCH_PALETTE[match_idx % len(MATCH_PALETTE)]
        self._set_status(f"Editing match #{match_idx + 1}")

    def _set_status(self, msg: str) -> None:
        self.status_msg = msg
        logger.info(f"  {msg}")

    # ------------------------------------------------------------------ #
    # Coordinate mapping
    # ------------------------------------------------------------------ #
    def _cam_from_x(self, x: int) -> tuple[int, int]:
        offset = 0
        for i in range(self.n_cams):
            if offset <= x < offset + self.panel_w[i]:
                return i, int((x - offset) / self.scales[i])
            offset += self.panel_w[i] + self.gap
        return -1, 0

    def _track_at(self, cam_idx: int, local_x: int, local_y: int):
        cam_name = self.cam_names[cam_idx]
        visible = tracks_io.get_boxes_at_frame(self.tracks[cam_name], self.frame_idx)
        # Prefer the smallest enclosing box so overlapping objects are pickable.
        hit = None
        hit_area = None
        for det in visible:
            x1, y1, x2, y2 = det["box"]
            if x1 <= local_x <= x2 and y1 <= local_y <= y2:
                area = (x2 - x1) * (y2 - y1)
                if hit_area is None or area < hit_area:
                    hit, hit_area = det, area
        return hit

    # ------------------------------------------------------------------ #
    # Mouse handling
    # ------------------------------------------------------------------ #
    def mouse_callback(self, event, x, y, flags, param):
        cam_idx, local_x = self._cam_from_x(x)
        if cam_idx < 0:
            self.hover = None
            return
        local_y = int(y / self.scales[cam_idx])

        if event == cv2.EVENT_MOUSEMOVE:
            det = self._track_at(cam_idx, local_x, local_y)
            self.hover = (cam_idx, det["track_id"]) if det else None
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            if self.pending_confirm:
                return  # ignore clicks while a confirm prompt is open
            det = self._track_at(cam_idx, local_x, local_y)
            if not det:
                return
            cam_name = self.cam_names[cam_idx]
            tid = det["track_id"]

            # Clicking a matched track (not already editing) -> enter edit mode.
            if (cam_name, tid) in self.matched_tracks and self.editing_match_idx is None:
                match_idx = self._find_match_for_track(cam_name, tid)
                if match_idx is not None:
                    self.selection_history.append(self.selected.copy())
                    self._load_match_into_selection(match_idx)
                    return

            self.selection_history.append(self.selected.copy())
            if self.selected.get(cam_name) == tid:
                del self.selected[cam_name]
                self._set_status(f"Deselected {cam_name} T{tid}")
            else:
                self.selected[cam_name] = tid
                self._set_status(
                    f"Selected {cam_name} T{tid} ({det['class_name']}, {det['conf']:.2f})"
                )

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def read_frames(self) -> list:
        frames = []
        for i, cap in enumerate(self.caps):
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_idx)
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((self.cameras[i].height or 480,
                                  self.cameras[i].width or 640, 3), np.uint8)
            frames.append(frame)
        return frames

    def draw_boxes(self, panel, cam_idx: int):
        cam_name = self.cam_names[cam_idx]
        scale = self.scales[cam_idx]
        visible = tracks_io.get_boxes_at_frame(self.tracks[cam_name], self.frame_idx)

        for det in visible:
            tid = det["track_id"]
            x1, y1, x2, y2 = [int(v * scale) for v in det["box"]]

            is_selected = self.selected.get(cam_name) == tid
            is_matched = (cam_name, tid) in self.matched_tracks
            is_hover = self.hover == (cam_idx, tid)

            if is_selected:
                color, thickness = self.current_select_color, 3
            elif is_matched:
                color, thickness = self.match_color_map.get((cam_name, tid), UNMATCHED_COLOR), 2
            else:
                color, thickness = UNMATCHED_COLOR, 2

            if is_hover:
                cv2.rectangle(panel, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), HOVER_COLOR, thickness + 1)
            cv2.rectangle(panel, (x1, y1), (x2, y2), color, thickness)

            # Label only matched / selected / hovered boxes to reduce clutter.
            if is_selected or is_matched or is_hover:
                label = f"T{tid} {det['class_name']}"
                lw, lh = measure_text(label, scale=0.5)
                cv2.rectangle(panel, (x1, y1 - lh - 10), (x1 + lw + 8, y1), color, -1)
                draw_text(panel, label, (x1 + 4, y1 - 5), scale=0.5, color=(0, 0, 0), weight=1)
        return panel

    def render(self):
        frames = self.read_frames()
        canvas = np.full((self.canvas_h, self.canvas_w, 3), 30, dtype=np.uint8)

        x_offset = 0
        for i, frame in enumerate(frames):
            panel = cv2.resize(frame, (self.panel_w[i], self.panel_h))
            panel = self.draw_boxes(panel, i)

            cam_color = CAM_COLORS[i % len(CAM_COLORS)]
            draw_text(panel, self.cam_names[i], (8, 26), scale=0.65, color=cam_color, weight=2)

            if self.cam_names[i] in self.selected:
                tid = self.selected[self.cam_names[i]]
                lbl = f"Selected: T{tid}"
                if self.editing_match_idx is not None:
                    lbl += " (EDITING)"
                draw_text(panel, lbl, (8, self.panel_h - 12), scale=0.6,
                          color=self.current_select_color, weight=2)
                cv2.rectangle(panel, (0, 0), (self.panel_w[i] - 1, self.panel_h - 1),
                              self.current_select_color, 3)

            canvas[0:self.panel_h, x_offset:x_offset + self.panel_w[i]] = panel
            x_offset += self.panel_w[i] + self.gap

        self._draw_info_bar(canvas)
        if self.show_help:
            self._draw_help(canvas)
        if self.pending_confirm:
            self._draw_confirm(canvas)
        return canvas

    def _draw_info_bar(self, canvas):
        y0 = self.panel_h
        # Solid dark band behind the text so it is always readable.
        cv2.rectangle(canvas, (0, y0), (self.canvas_w, self.canvas_h), (20, 20, 20), -1)
        cv2.line(canvas, (0, y0), (self.canvas_w, y0), (70, 70, 70), 1)

        elapsed = self.frame_idx / self.fps if self.fps else 0
        info = (
            f"Frame {self.frame_idx}/{self.total_frames - 1} | "
            f"Time {elapsed:.1f}s | Matches {len(self.matches)} | "
            f"Selected {len(self.selected)}/{self.n_cams}"
        )
        draw_text(canvas, info, (10, y0 + 26), scale=0.6, color=(255, 255, 255), weight=1)

        if self.selected:
            sel_text = " | ".join(f"{cn}:T{tid}" for cn, tid in self.selected.items())
            mode = "EDITING" if self.editing_match_idx is not None else "Linking"
            draw_text(canvas, f"{mode}: {sel_text}", (10, y0 + 52), scale=0.55,
                      color=self.current_select_color, weight=1)
        elif self.status_msg:
            draw_text(canvas, self.status_msg, (10, y0 + 52), scale=0.55,
                      color=(0, 220, 255), weight=1)

        state = "PAUSED" if self.paused else "PLAYING"
        draw_text(
            canvas,
            f"{state} | ENTER confirm  N next  D delete  Ctrl+Z undo  H help  Q quit",
            (10, y0 + 80), scale=0.5, color=(180, 180, 180), weight=1,
        )

    def _draw_help(self, canvas):
        pad = 16
        line_h = 24
        box_w = 430
        box_h = pad * 2 + line_h * len(HELP_LINES)
        x0 = (self.canvas_w - box_w) // 2
        y0 = max(10, (self.canvas_h - box_h) // 2)
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)
        cv2.rectangle(canvas, (x0, y0), (x0 + box_w, y0 + box_h), (90, 90, 90), 1)
        for i, line in enumerate(HELP_LINES):
            y = y0 + pad + line_h * (i + 1) - 6
            color = (0, 220, 255) if i == 0 else (235, 235, 235)
            weight = 2 if i == 0 else 1
            draw_text(canvas, line, (x0 + pad, y), scale=0.5, color=color, weight=weight)

    def _draw_confirm(self, canvas):
        _, message = self.pending_confirm
        box_w, box_h = 540, 110
        x0 = (self.canvas_w - box_w) // 2
        y0 = (self.canvas_h - box_h) // 2
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (10, 10, 40), -1)
        cv2.addWeighted(overlay, 0.9, canvas, 0.1, 0, canvas)
        cv2.rectangle(canvas, (x0, y0), (x0 + box_w, y0 + box_h), (0, 165, 255), 2)
        draw_text(canvas, message, (x0 + 20, y0 + 45), scale=0.6, color=(255, 255, 255), weight=1)
        draw_text(canvas, "Press Y to confirm, any other key to cancel",
                  (x0 + 20, y0 + 80), scale=0.55, color=(0, 220, 255), weight=1)

    # ------------------------------------------------------------------ #
    # Actions
    # ------------------------------------------------------------------ #
    def confirm_match(self):
        if len(self.selected) < 2:
            self._set_status("Need at least 2 cameras selected to make a match")
            return

        if self.editing_match_idx is not None and self.editing_match_idx < len(self.matches):
            idx = self.editing_match_idx
            old_match = _deep_copy_match(self.matches[idx])
            new_tracks = dict(self._match_tracks(self.matches[idx]))
            for cam_name, tid in self.selected.items():
                new_tracks[cam_name] = tid
            self.matches[idx] = {"frame": self.matches[idx].get("frame", self.frame_idx),
                                 "tracks": new_tracks}
            self.undo_stack.append(("edit", idx, old_match, _deep_copy_match(self.matches[idx])))
            self._set_status(f"Updated match #{idx + 1}")
        else:
            match = {"frame": self.frame_idx, "tracks": dict(self.selected)}
            self.matches.append(match)
            self.undo_stack.append(("add", _deep_copy_match(match)))
            self._set_status(f"Match #{len(self.matches)} confirmed")

        self.redo_stack.clear()
        self.editing_match_idx = None
        self._refresh_after_change()
        self.selected = {}
        self.selection_history = []
        self.save()

    def delete_last_match(self):
        if not self.matches:
            self._set_status("No matches to delete")
            return
        removed = self.matches.pop()
        self.undo_stack.append(("delete", _deep_copy_match(removed)))
        self.redo_stack.clear()
        self._clear_selection()
        self._refresh_after_change()
        self._set_status("Deleted last match")
        self.save()

    def delete_match(self, idx: int):
        """Delete a specific match by index (used when deleting the edited one)."""
        if idx is None or idx < 0 or idx >= len(self.matches):
            return self.delete_last_match()
        removed = self.matches.pop(idx)
        self.undo_stack.append(("delete_at", idx, _deep_copy_match(removed)))
        self.redo_stack.clear()
        self._clear_selection()
        self._refresh_after_change()
        self._set_status(f"Deleted match #{idx + 1}")
        self.save()

    def clear_all_matches(self):
        if not self.matches:
            return
        snapshot = [_deep_copy_match(m) for m in self.matches]
        self.undo_stack.append(("clear_all", snapshot))
        self.redo_stack.clear()
        self.matches = []
        self._clear_selection()
        self._refresh_after_change()
        self._set_status("Cleared ALL matches")
        self.save()

    def _clear_selection(self):
        """Drop the current selection / edit state so the video deselects."""
        self.selected = {}
        self.selection_history = []
        self.editing_match_idx = None

    def undo(self):
        if not self.undo_stack:
            self._set_status("Nothing to undo")
            return
        item = self.undo_stack.pop()
        action = item[0]
        if action == "add":
            data = item[1]
            self._remove_match(data)
            self.redo_stack.append(("add", data))
            self._set_status("Undo: removed match")
        elif action == "delete":
            data = item[1]
            self.matches.append(data)
            self.redo_stack.append(("delete", data))
            self._set_status("Undo: restored match")
        elif action == "delete_at":
            idx, data = item[1], item[2]
            self.matches.insert(min(idx, len(self.matches)), data)
            self.redo_stack.append(("delete_at", idx, data))
            self._set_status(f"Undo: restored match #{idx + 1}")
        elif action == "edit":
            idx, old_match, new_match = item[1], item[2], item[3]
            if idx < len(self.matches):
                self.matches[idx] = old_match
            self.redo_stack.append(("edit", idx, old_match, new_match))
            self._set_status(f"Undo: reverted edit on match #{idx + 1}")
        elif action == "clear_all":
            snapshot = item[1]
            self.matches = [_deep_copy_match(m) for m in snapshot]
            self.redo_stack.append(("clear_all", snapshot))
            self._set_status("Undo: restored all matches")
        self._refresh_after_change()
        self.save()

    def redo(self):
        if not self.redo_stack:
            self._set_status("Nothing to redo")
            return
        item = self.redo_stack.pop()
        action = item[0]
        if action == "add":
            data = item[1]
            self.matches.append(data)
            self.undo_stack.append(("add", data))
            self._set_status("Redo: re-added match")
        elif action == "delete":
            data = item[1]
            self._remove_match(data)
            self.undo_stack.append(("delete", data))
            self._set_status("Redo: re-deleted match")
        elif action == "delete_at":
            idx, data = item[1], item[2]
            if idx < len(self.matches):
                self.matches.pop(idx)
            else:
                self._remove_match(data)
            self.undo_stack.append(("delete_at", idx, data))
            self._set_status(f"Redo: re-deleted match #{idx + 1}")
        elif action == "edit":
            idx, old_match, new_match = item[1], item[2], item[3]
            if idx < len(self.matches):
                self.matches[idx] = new_match
            self.undo_stack.append(("edit", idx, old_match, new_match))
            self._set_status(f"Redo: re-applied edit on match #{idx + 1}")
        elif action == "clear_all":
            snapshot = item[1]
            self.undo_stack.append(("clear_all", snapshot))
            self.matches = []
            self._set_status("Redo: cleared all matches")
        self._refresh_after_change()
        self.save()

    def _remove_match(self, data: dict):
        for i, m in enumerate(self.matches):
            if m is data or m == data:
                self.matches.pop(i)
                return

    def _refresh_after_change(self):
        self.matched_tracks = self._build_matched_set()
        self.match_color_map = self._build_color_map()
        self.current_select_color = self._next_color()

    def find_next_unmatched(self):
        ref_cam = self.cam_names[0]
        for tid, track in self.tracks[ref_cam].items():
            if (ref_cam, int(tid)) not in self.matched_tracks and track["frames"]:
                mid = len(track["frames"]) // 2
                self.frame_idx = track["frames"][mid]
                self.selected = {ref_cam: int(tid)}
                self.selection_history = []
                self._set_status(f"Jumped to unmatched {ref_cam} T{tid}")
                return
        self._set_status(f"No more unmatched tracks in '{ref_cam}'")

    def save(self):
        tracks_io.save_matches(self.project.matches_path, self.matches)

    def print_matches(self):
        print(f"\n  --- Matches ({len(self.matches)}) ---")
        for i, m in enumerate(self.matches):
            tracks_str = ", ".join(
                f"{cn}:T{self._match_tracks(m).get(cn, '-')}" for cn in self.cam_names
            )
            print(f"    #{i + 1}: frame {m.get('frame', '?')} -> {tracks_str}")
        print()

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def run(self):
        window = f"multicam_reid matcher - {self.project.folder.name}"
        # WINDOW_GUI_NORMAL removes OpenCV's native Qt toolbar (zoom/pan/save),
        # whose blurry tooltips are unreadable; we provide our own H help instead.
        cv2.namedWindow(window, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(window, min(self.canvas_w, MAX_CANVAS_WIDTH),
                         min(self.canvas_h, MAX_CANVAS_HEIGHT))
        cv2.setMouseCallback(window, self.mouse_callback)

        print(f"\n  Cross-camera matcher - {self.project.folder}")
        print(f"  {'=' * 52}")
        print(f"  Cameras: {', '.join(self.cam_names)}")
        print(f"  Existing matches: {len(self.matches)}")
        print(f"  Press H in the window for the full controls list.\n")

        while True:
            cv2.imshow(window, self.render())
            wait_ms = 0 if self.paused else max(1, int(1000 / self.fps))
            key = cv2.waitKeyEx(wait_ms)

            if key == -1:
                if not self.paused:
                    self.frame_idx = min(self.frame_idx + 1, self.total_frames - 1)
                continue

            # Confirmation prompt intercepts the next keystroke.
            if self.pending_confirm:
                action, _ = self.pending_confirm
                self.pending_confirm = None
                if key in (ord('y'), ord('Y')):
                    if action == "delete_last":
                        self.delete_last_match()
                    elif action == "delete_current":
                        self.delete_match(self._pending_delete_idx)
                    elif action == "clear_all":
                        self.clear_all_matches()
                else:
                    self._set_status("Cancelled")
                continue

            if key in (ord('q'), 27):  # Q / ESC
                self.save()
                logger.info(f"  Saved {len(self.matches)} matches to {self.project.matches_path}")
                break
            elif key == 13:  # ENTER
                self.confirm_match()
            elif key in (8, 127):  # Backspace / Delete
                self._clear_selection()
                self.current_select_color = self._next_color()
                self._set_status("Selection cleared")
            elif key == ord('n'):
                self.find_next_unmatched()
            elif key == ord('d'):
                if self.editing_match_idx is not None:
                    self._pending_delete_idx = self.editing_match_idx
                    self.pending_confirm = ("delete_current",
                                            f"Delete match #{self.editing_match_idx + 1} "
                                            f"(the one being edited)?")
                elif self.matches:
                    self.pending_confirm = ("delete_last", "Delete the last match?")
            elif key == ord('x'):
                if self.matches:
                    self.pending_confirm = ("clear_all",
                                            f"Clear ALL {len(self.matches)} matches?")
            elif key == 26:  # Ctrl+Z
                if self.selected:
                    if self.selection_history:
                        self.selected = self.selection_history.pop()
                        if not self.selected:
                            self.editing_match_idx = None
                        self._set_status("Undo selection")
                    else:
                        self._clear_selection()
                        self._set_status("Selection cleared")
                else:
                    self.undo()
            elif key == 25:  # Ctrl+Y
                self.redo()
            elif key == ord('h'):
                self.show_help = not self.show_help
            elif key == ord(' '):
                self.paused = not self.paused
            elif key == ord('.'):
                self.frame_idx = min(self.frame_idx + 1, self.total_frames - 1)
            elif key == ord(','):
                self.frame_idx = max(self.frame_idx - 1, 0)
            elif key == ord('w'):
                self.frame_idx = min(self.frame_idx + int(5 * self.fps), self.total_frames - 1)
            elif key == ord('s'):
                self.frame_idx = max(self.frame_idx - int(5 * self.fps), 0)
            elif key == 65363:  # Right arrow
                self.frame_idx = min(self.frame_idx + 10, self.total_frames - 1)
            elif key == 65361:  # Left arrow
                self.frame_idx = max(self.frame_idx - 10, 0)
            elif key == 9:  # TAB
                self.print_matches()
            elif not self.paused:
                self.frame_idx = min(self.frame_idx + 1, self.total_frames - 1)

        for cap in self.caps:
            cap.release()
        cv2.destroyAllWindows()


def _deep_copy_match(m: dict) -> dict:
    """Copy a match dict including its nested tracks mapping."""
    return {"frame": m.get("frame"), "tracks": dict(m.get("tracks", {}))}
