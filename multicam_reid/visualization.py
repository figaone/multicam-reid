"""
PRIME Set 12 Visualization - Tracked trajectories with global IDs
Generates an annotated video showing tracked bounding boxes and cross-camera match IDs.
"""
import json
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from multicam_reid.ui.render import draw_text, measure_text


def load_json(path):
    with open(path) as f:
        return json.load(f)


def get_color_for_id(match_id):
    """Get consistent color for a global match ID."""
    np.random.seed(match_id % 256)
    return tuple(np.random.randint(50, 255, 3).tolist())


def visualize_matches(workspace_dir, output_video, max_frames=300):
    """Create annotated video with tracked boxes and global IDs."""
    ws = Path(workspace_dir)
    manifest = load_json(ws / "manifest.json")
    matches_data = load_json(ws / "matches.json")
    matches = matches_data.get("matches", [])
    
    cameras = {cam["name"]: cam for cam in manifest["cameras"]}
    
    # Load tracks
    tracks_by_camera = {}
    for cam_name in cameras:
        track_file = ws / "tracks" / f"{cam_name}.tracks.json"
        if track_file.exists():
            data = load_json(track_file)
            tracks_by_camera[cam_name] = data
    
    # Build reverse mapping: (camera, track_id) -> match_id
    track_to_match_id = {}
    for match_idx, match in enumerate(matches):
        match_id = match_idx + 1
        for cam_name, track_id in match["tracks"].items():
            track_to_match_id[(cam_name, track_id)] = match_id
    
    # Open videos
    video_files = {}
    for cam_name, cam_info in cameras.items():
        video_path = ws.parent / cam_info["video"]
        if video_path.exists():
            video_files[cam_name] = cv2.VideoCapture(str(video_path))
    
    if not video_files:
        print("ERROR: No video files found")
        return
    
    # Build per-camera output sizes so each box uses the correct scale.
    panel_h = 600
    cam_order = [name for name in cameras.keys() if name in video_files]
    panel_w = {}
    panel_scale = {}
    x_offsets = {}

    x_cursor = 0
    for cam_name in cam_order:
        cam = cameras[cam_name]
        in_w = max(1, int(cam["width"]))
        in_h = max(1, int(cam["height"]))
        out_w = int(round(in_w * (panel_h / in_h)))
        panel_w[cam_name] = out_w
        panel_scale[cam_name] = (out_w / in_w, panel_h / in_h)
        x_offsets[cam_name] = x_cursor
        x_cursor += out_w

    grid_w, grid_h = x_cursor, panel_h
    out_fps = 10.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video), fourcc, out_fps, (grid_w, grid_h))
    
    print(f"Output grid: {grid_w}x{grid_h} @ {out_fps}fps")
    print(f"Processing up to {max_frames} frames...")
    
    # Process frames
    frame_idx = 0
    while frame_idx < max_frames:
        frames_read = {}
        for cam_name in cam_order:
            cap = video_files[cam_name]
            ret, frame = cap.read()
            if ret:
                frames_read[cam_name] = cv2.resize(frame, (panel_w[cam_name], panel_h))
            else:
                break
        
        if not frames_read or len(frames_read) < len(video_files):
            break
        
        # Create grid
        grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        
        for cam_name in cam_order:
            frame = frames_read[cam_name]
            x_off = x_offsets[cam_name]
            grid[:, x_off:x_off + panel_w[cam_name]] = frame
            
            # Draw tracks for this camera
            if cam_name in tracks_by_camera:
                tracks = tracks_by_camera[cam_name]
                for track_id_str, track in tracks.items():
                    track_id = int(track_id_str)
                    frames = track.get("frames", [])
                    boxes = track.get("boxes", [])
                    
                    # Check if this frame is in the track
                    if frame_idx in frames:
                        frame_pos = frames.index(frame_idx)
                        box = boxes[frame_pos]
                        sx, sy = panel_scale[cam_name]
                        x1 = int(box[0] * sx)
                        y1 = int(box[1] * sy)
                        x2 = int(box[2] * sx)
                        y2 = int(box[3] * sy)
                        
                        # Check if this track is in a match
                        match_id = track_to_match_id.get((cam_name, track_id))
                        if match_id:
                            color = get_color_for_id(match_id)
                            thickness = 3
                            text = f"ID:{match_id}"
                        else:
                            color = (100, 100, 100)  # Gray for unmatched
                            thickness = 1
                            text = f"T:{track_id}"
                        
                        # Draw box
                        cv2.rectangle(grid, (x_off + x1, y1), (x_off + x2, y2), color, thickness)
                        # Draw ID
                        cv2.putText(grid, text, (x_off + x1, y1 - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Add frame counter and camera labels
        for cam_name in cam_order:
            label = cam_name.split("_synced")[0] if "_synced" in cam_name else cam_name
            x_off = x_offsets[cam_name]
            cv2.putText(grid, f"{label} | Frame {frame_idx}",
                       (x_off + 5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        writer.write(grid)
        frame_idx += 1
        
        if frame_idx % 50 == 0:
            print(f"  Frame {frame_idx}...")
    
    writer.release()
    for cap in video_files.values():
        cap.release()
    
    print(f"Visualization saved: {output_video}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -c 'from multicam_reid.visualization import visualize_matches; visualize_matches(...)'")
        sys.exit(1)
    
    workspace = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else "matched_tracks.mp4"
    max_frames = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    
    visualize_matches(workspace, output, max_frames)
