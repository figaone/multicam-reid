"""
PRIME Set 12 Evaluation Metrics
Analyze tracking quality, cross-camera association, and matching performance.
"""
import json
from pathlib import Path
from collections import defaultdict
import statistics


def load_json(path):
    """Load JSON file."""
    with open(path) as f:
        return json.load(f)


def compute_metrics(workspace_dir):
    """Compute tracking and matching metrics for a segment."""
    ws = Path(workspace_dir)
    
    # Load manifests and tracks
    manifest = load_json(ws / "manifest.json")
    matches_data = load_json(ws / "matches.json")
    matches = matches_data.get("matches", [])
    
    cameras = {cam["name"]: cam for cam in manifest["cameras"]}
    
    # Load tracks per camera (keyed by track_id, not wrapped in "tracks" list)
    tracks_by_camera = {}
    for cam_name in cameras:
        track_file = ws / "tracks" / f"{cam_name}.tracks.json"
        if track_file.exists():
            data = load_json(track_file)
            # data is {track_id: {frames, boxes}, ...}
            tracks_by_camera[cam_name] = list(data.values())
    
    # === Basic Stats ===
    print("\n" + "="*70)
    print("SET 12 EVALUATION METRICS")
    print("="*70)
    
    print("\n[1] CAMERA COVERAGE & TRACKING")
    print("-" * 70)
    total_tracks = 0
    per_cam_stats = {}
    
    for cam_name, cam_info in cameras.items():
        tracks = tracks_by_camera.get(cam_name, [])
        n_tracks = len(tracks)
        total_tracks += n_tracks
        
        # Track duration stats
        track_lengths = [len(t.get("boxes", [])) for t in tracks]
        if track_lengths:
            avg_len = statistics.mean(track_lengths)
            min_len = min(track_lengths)
            max_len = max(track_lengths)
        else:
            avg_len = min_len = max_len = 0
        
        per_cam_stats[cam_name] = {
            "tracks": n_tracks,
            "frames": cam_info["frame_count"],
            "fps": cam_info["fps"],
            "duration_sec": cam_info["frame_count"] / cam_info["fps"],
            "avg_track_len": avg_len,
            "min_track_len": min_len,
            "max_track_len": max_len,
        }
        
        print(f"  {cam_name}")
        print(f"    Resolution: {cam_info['width']}x{cam_info['height']}")
        print(f"    Duration: {cam_info['frame_count']} frames @ {cam_info['fps']:.1f}fps = {cam_info['frame_count']/cam_info['fps']:.1f}s")
        print(f"    Tracks: {n_tracks}")
        print(f"    Track length: avg={avg_len:.1f}, min={min_len}, max={max_len}")
    
    print(f"\n  Total tracks across all cameras: {total_tracks}")
    
    # === Cross-Camera Association ===
    print("\n[2] CROSS-CAMERA ASSOCIATION (ReID Matching)")
    print("-" * 70)
    
    n_matched_ids = len(matches)
    print(f"  Global matched IDs: {n_matched_ids}")
    
    # Distribution across cameras
    match_coverage = defaultdict(int)
    for match in matches:
        for cam_name in match["tracks"]:
            match_coverage[cam_name] += 1
    
    print(f"\n  Matches per camera:")
    for cam_name in sorted(match_coverage.keys()):
        count = match_coverage[cam_name]
        total = per_cam_stats[cam_name]["tracks"]
        pct = 100 * count / total if total > 0 else 0
        print(f"    {cam_name}: {count}/{total} ({pct:.1f}%)")
    
    # Matching patterns (how many cameras does each global ID appear in)
    cameras_per_id = defaultdict(int)
    for match in matches:
        cameras_per_id[len(match["tracks"])] += 1
    
    print(f"\n  Global ID camera coverage:")
    for n_cams in sorted(cameras_per_id.keys()):
        count = cameras_per_id[n_cams]
        pct = 100 * count / n_matched_ids if n_matched_ids > 0 else 0
        print(f"    {n_cams} camera(s): {count} IDs ({pct:.1f}%)")
    
    # === Quality Metrics ===
    print("\n[3] MATCHING QUALITY")
    print("-" * 70)
    
    # Check track fragmentation
    matched_tracks = defaultdict(int)
    for match in matches:
        for cam_name, track_id in match["tracks"].items():
            # track_id is a single integer (track ID in that camera)
            matched_tracks[cam_name] += 1
    
    print(f"  Tracks in global matches per camera:")
    for cam_name in sorted(per_cam_stats.keys()):
        total = per_cam_stats[cam_name]["tracks"]
        matched = matched_tracks.get(cam_name, 0)
        unmatched = total - matched
        pct = 100 * matched / total if total > 0 else 0
        print(f"    {cam_name}: {matched}/{total} ({pct:.1f}%) matched")
        if unmatched > 0:
            print(f"      → {unmatched} unmatched (single-camera only or low confidence)")
    
    # === Summary ===
    print("\n[4] SUMMARY")
    print("-" * 70)
    avg_track_len = statistics.mean([s["avg_track_len"] for s in per_cam_stats.values()])
    print(f"  Total detections: ~{sum(t['tracks'] * t['avg_track_len'] for t in per_cam_stats.values()):.0f}")
    print(f"  Tracking efficiency: {n_matched_ids}/{total_tracks} IDs across cameras")
    print(f"  Cross-camera association rate: {100*n_matched_ids/total_tracks:.1f}%")
    print(f"  Average track duration: {avg_track_len:.1f} frames ({avg_track_len/10:.1f}s @ 10fps)")
    
    print("\n" + "="*70 + "\n")
    
    return {
        "per_camera": per_cam_stats,
        "total_tracks": total_tracks,
        "matched_ids": n_matched_ids,
        "matched_track_coverage": matched_tracks,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m multicam_reid.evaluation <workspace_dir>")
        sys.exit(1)
    
    compute_metrics(sys.argv[1])
