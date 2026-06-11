"""
Project / workspace management.

A "project" is simply a folder that contains video files (one per camera).
The toolkit creates a hidden `.reid/` subfolder inside it to store all
derived data, so a user only ever needs to point at their own folder:

    my_intersection/
    |- cam_north.mp4          <- user's videos (any names)
    |- cam_east.mp4
    |- cam_west.mp4
    `- .reid/                 <- auto-created workspace
       |- manifest.json       <- cameras, paths, fps, frame counts, offsets
       |- tracks/
       |  |- cam_north.tracks.json
       |  |- cam_east.tracks.json
       |  `- cam_west.tracks.json
       `- matches.json        <- cross-camera ground-truth matches

Re-opening the same folder loads everything back automatically.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
from loguru import logger

WORKSPACE_DIRNAME = ".reid"
MANIFEST_NAME = "manifest.json"
TRACKS_DIRNAME = "tracks"
MATCHES_NAME = "matches.json"
MANIFEST_VERSION = 1

# Video extensions we will auto-discover inside a project folder.
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".mpg", ".mpeg", ".wmv"}


@dataclass
class Camera:
    """A single camera in the project."""

    name: str                 # stable identifier (derived from filename)
    video: str                # path relative to the project folder
    width: int = 0
    height: int = 0
    fps: float = 0.0
    frame_count: int = 0
    frame_offset: int = 0     # optional sync offset (frames added before this cam)

    @property
    def tracks_filename(self) -> str:
        return f"{self.name}.tracks.json"


@dataclass
class Manifest:
    """Describes a project's cameras and metadata."""

    version: int = MANIFEST_VERSION
    cameras: list[Camera] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "cameras": [asdict(c) for c in self.cameras],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Manifest":
        cams = [Camera(**c) for c in data.get("cameras", [])]
        return cls(version=data.get("version", MANIFEST_VERSION), cameras=cams)


class Project:
    """Manages a project folder and its `.reid/` workspace."""

    def __init__(self, folder: str | Path):
        self.folder = Path(folder).expanduser().resolve()
        if not self.folder.exists():
            raise FileNotFoundError(f"Project folder not found: {self.folder}")
        if not self.folder.is_dir():
            raise NotADirectoryError(f"Project path is not a folder: {self.folder}")

        self.workspace = self.folder / WORKSPACE_DIRNAME
        self.tracks_dir = self.workspace / TRACKS_DIRNAME
        self.manifest_path = self.workspace / MANIFEST_NAME
        self.matches_path = self.workspace / MATCHES_NAME
        self.manifest: Manifest = Manifest()

    # ------------------------------------------------------------------ #
    # Workspace lifecycle
    # ------------------------------------------------------------------ #
    def exists(self) -> bool:
        """True if this folder already has an initialized workspace."""
        return self.manifest_path.exists()

    def ensure_workspace(self) -> None:
        """Create the `.reid/` workspace directories if missing."""
        self.tracks_dir.mkdir(parents=True, exist_ok=True)

    def discover_videos(self) -> list[Path]:
        """Find candidate video files directly inside the project folder."""
        videos = sorted(
            p for p in self.folder.iterdir()
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        )
        return videos

    def _probe_video(self, path: Path) -> tuple[int, int, float, int]:
        """Read width, height, fps, and frame count from a video file."""
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            cap.release()
            raise IOError(f"Could not open video: {path}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return width, height, fps, frame_count

    def init(self, force: bool = False) -> Manifest:
        """
        Initialize (or re-initialize) the workspace by discovering videos
        and probing their metadata. Returns the resulting manifest.
        """
        if self.exists() and not force:
            return self.load()

        videos = self.discover_videos()
        if len(videos) < 2:
            raise ValueError(
                f"Need at least 2 videos in {self.folder} to match across "
                f"cameras (found {len(videos)}). Supported extensions: "
                f"{', '.join(sorted(VIDEO_EXTENSIONS))}"
            )

        self.ensure_workspace()
        cameras: list[Camera] = []
        used_names: set[str] = set()
        for video in videos:
            name = _safe_name(video.stem, used_names)
            used_names.add(name)
            width, height, fps, frame_count = self._probe_video(video)
            cameras.append(
                Camera(
                    name=name,
                    video=video.name,
                    width=width,
                    height=height,
                    fps=fps,
                    frame_count=frame_count,
                )
            )
            logger.info(
                f"  Camera '{name}': {video.name} "
                f"({width}x{height}, {fps:.1f}fps, {frame_count} frames)"
            )

        self.manifest = Manifest(cameras=cameras)
        self.save_manifest()
        logger.info(f"  Initialized workspace with {len(cameras)} cameras")
        return self.manifest

    def load(self) -> Manifest:
        """Load an existing manifest from disk."""
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"No workspace found in {self.folder}. Run 'track' or 'init' first."
            )
        with open(self.manifest_path) as f:
            data = json.load(f)
        self.manifest = Manifest.from_dict(data)
        return self.manifest

    def save_manifest(self) -> None:
        """Persist the manifest to disk."""
        self.ensure_workspace()
        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest.to_dict(), f, indent=2)

    # ------------------------------------------------------------------ #
    # Convenience accessors
    # ------------------------------------------------------------------ #
    @property
    def cameras(self) -> list[Camera]:
        return self.manifest.cameras

    @property
    def camera_names(self) -> list[str]:
        return [c.name for c in self.manifest.cameras]

    def video_path(self, cam: Camera) -> Path:
        return self.folder / cam.video

    def tracks_path(self, cam: Camera) -> Path:
        return self.tracks_dir / cam.tracks_filename

    def has_tracks(self) -> bool:
        """True if every camera already has a cached tracks file."""
        if not self.manifest.cameras:
            return False
        return all(self.tracks_path(c).exists() for c in self.manifest.cameras)

    def missing_tracks(self) -> list[Camera]:
        """Return cameras that do not yet have cached tracks."""
        return [c for c in self.manifest.cameras if not self.tracks_path(c).exists()]


def _safe_name(stem: str, used: set[str]) -> str:
    """Make a filesystem/JSON-safe, unique camera name from a filename stem."""
    cleaned = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in stem)
    cleaned = cleaned.strip("_") or "camera"
    name = cleaned
    counter = 2
    while name in used:
        name = f"{cleaned}_{counter}"
        counter += 1
    return name
