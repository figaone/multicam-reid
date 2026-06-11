"""
multicam_reid command-line interface.

Commands:
    init    <folder>            Discover videos and create the workspace.
    sync    <folder>            Manually align cameras and export synced segments.
    track   <folder>            Run detection + tracking (caches results).
    match   <folder>            Launch the interactive cross-camera matcher.
    export  <folder>            Export a ReID crop dataset from matches.
    info    <folder>            Print project status.

Typical workflow:
    python -m multicam_reid sync   /path/to/raw_videos   # (optional) align + cut segments
    python -m multicam_reid track  /path/to/my_videos
    python -m multicam_reid match  /path/to/my_videos
    python -m multicam_reid export /path/to/my_videos

The matcher also auto-runs tracking for you if no cached tracks are found,
so for the simplest path you can just run:
    python -m multicam_reid match /path/to/my_videos
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

from .core.project import Project


def _confirm(prompt: str, default_yes: bool = True) -> bool:
    suffix = "[Y/n]" if default_yes else "[y/N]"
    try:
        answer = input(f"{prompt} {suffix} ").strip().lower()
    except EOFError:
        return default_yes
    if not answer:
        return default_yes
    return answer in ("y", "yes")


def _ensure_project(folder: str, force: bool = False) -> Project:
    project = Project(folder)
    if project.exists() and not force:
        project.load()
    else:
        project.init(force=force)
    return project


def _run_tracking(project: Project, args) -> None:
    from .core.tracker import load_model, run_tracker
    from .core import tracks_io

    if args.force:
        targets = project.cameras
    else:
        targets = project.missing_tracks()
        if not targets:
            logger.info("  All cameras already have cached tracks (use --force to redo).")
            return

    model = load_model(args.model)
    for cam in targets:
        tracks = run_tracker(
            project.video_path(cam),
            model,
            conf=args.conf,
            tracker=args.tracker,
        )
        tracks_io.save_tracks(project.tracks_path(cam), tracks)
        logger.info(f"  Saved tracks for '{cam.name}' -> {project.tracks_path(cam).name}")


def cmd_init(args):
    project = Project(args.folder)
    project.init(force=args.force)
    logger.info(f"  Workspace ready at {project.workspace}")
    cmd_info(args, project=project)


def cmd_track(args):
    project = _ensure_project(args.folder, force=False)
    _run_tracking(project, args)


def cmd_match(args):
    project = _ensure_project(args.folder, force=False)

    if not project.has_tracks():
        missing = [c.name for c in project.missing_tracks()]
        logger.warning(f"  No cached tracks for: {', '.join(missing)}")
        if _confirm("  Run detection + tracking now?"):
            _run_tracking(project, args)
        else:
            logger.error("  Cannot launch matcher without tracks. Run 'track' first.")
            return

    from .ui.matcher import Matcher

    matcher = Matcher(project, start_frame=args.frame)
    matcher.run()


def cmd_export(args):
    project = _ensure_project(args.folder, force=False)
    from .core.exporter import export_reid_dataset

    export_reid_dataset(
        project,
        samples_per_camera=args.samples,
        min_box_size=args.min_size,
        padding=args.padding,
    )


def cmd_sync(args):
    project = _ensure_project(args.folder, force=False)
    from .ui.sync_tool import SyncTool

    tool = SyncTool(project)
    tool.run()


def cmd_info(args, project: Project | None = None):
    project = project or _ensure_project(args.folder, force=False)
    from .core import tracks_io

    print(f"\n  Project: {project.folder}")
    print(f"  Workspace: {project.workspace}")
    print(f"  Cameras ({len(project.cameras)}):")
    for cam in project.cameras:
        has = project.tracks_path(cam).exists()
        n_tracks = len(tracks_io.load_tracks(project.tracks_path(cam))) if has else 0
        status = f"{n_tracks} tracks" if has else "no tracks (run 'track')"
        print(f"    - {cam.name:20s} {cam.width}x{cam.height} "
              f"{cam.fps:.1f}fps {cam.frame_count}f  [{status}]")
    matches = tracks_io.load_matches(project.matches_path)
    print(f"  Matches: {len(matches)}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="multicam_reid",
        description="Cross-camera vehicle re-identification annotation toolkit",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_tracking_opts(p):
        p.add_argument("--model", default="yolov8x.pt", help="YOLO model (default: yolov8x.pt)")
        p.add_argument("--conf", type=float, default=0.3, help="Detection confidence threshold")
        p.add_argument("--tracker", default="bytetrack.yaml",
                       help="Tracker config (bytetrack.yaml / botsort.yaml)")
        p.add_argument("--force", action="store_true", help="Re-run even if tracks exist")

    p_init = sub.add_parser("init", help="Discover videos and create the workspace")
    p_init.add_argument("folder")
    p_init.add_argument("--force", action="store_true", help="Re-initialize the manifest")
    p_init.set_defaults(func=cmd_init)

    p_track = sub.add_parser("track", help="Run detection + tracking")
    p_track.add_argument("folder")
    add_tracking_opts(p_track)
    p_track.set_defaults(func=cmd_track)

    p_sync = sub.add_parser("sync", help="Manually align cameras and export segments")
    p_sync.add_argument("folder")
    p_sync.set_defaults(func=cmd_sync)

    p_match = sub.add_parser("match", help="Launch the interactive matcher")
    p_match.add_argument("folder")
    p_match.add_argument("--frame", type=int, default=0, help="Starting frame")
    add_tracking_opts(p_match)
    p_match.set_defaults(func=cmd_match)

    p_export = sub.add_parser("export", help="Export a ReID crop dataset from matches")
    p_export.add_argument("folder")
    p_export.add_argument("--samples", type=int, default=5,
                          help="Frames cropped per camera per ID")
    p_export.add_argument("--min-size", type=int, default=16,
                          help="Minimum crop size in pixels")
    p_export.add_argument("--padding", type=float, default=0.0,
                          help="Fractional padding around boxes (0.1 = +10%%)")
    p_export.set_defaults(func=cmd_export)

    p_info = sub.add_parser("info", help="Print project status")
    p_info.add_argument("folder")
    p_info.set_defaults(func=cmd_info)

    return parser


def main(argv: list[str] | None = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (FileNotFoundError, NotADirectoryError, ValueError, IOError) as exc:
        logger.error(f"  {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
