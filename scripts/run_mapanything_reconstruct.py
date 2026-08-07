"""
Run MapAnything on the 3 fixed camera stills to recover a shared metric 3D
reconstruction. Saves one .npz per camera for the adapter to turn into a shared
ground frame.

IMPORTANT: run this in the MapAnything conda env (NOT the project .venv):
    conda activate mapanything
    python scripts/run_mapanything_reconstruct.py \
        --images cam1.png cam2.png cam3.png \
        --cam-nums 1 2 3 \
        --out-dir /path/to/recon --apache

Each still should be the SAME synchronized frame index from each camera, ideally
a moment with the road/landmarks clearly visible (few vehicles is better -- we
only need the static scene geometry).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def to_np(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().float().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def squeeze_batch(a):
    a = to_np(a)
    if a.ndim >= 1 and a.shape[0] == 1:
        a = a[0]
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True, help="Stills, one per camera, in order.")
    ap.add_argument("--cam-nums", nargs="+", type=int, required=True, help="Camera numbers matching --images.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--apache", action="store_true", help="Use the Apache-2.0 licensed model.")
    args = ap.parse_args()

    assert len(args.images) == len(args.cam_nums), "images and cam-nums must match in length"

    import torch
    from PIL import Image
    from mapanything.models import MapAnything
    from mapanything.utils.image import load_images

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = "facebook/map-anything-apache" if args.apache else "facebook/map-anything"
    print(f"Loading {ckpt} on {device} ...")
    model = MapAnything.from_pretrained(ckpt).to(device)

    views = load_images(list(args.images))
    print(f"Running inference on {len(views)} views ...")
    use_amp = device == "cuda"   # autocast/bf16 only helps on GPU
    preds = model.infer(
        views,
        memory_efficient_inference=True,
        use_amp=use_amp,
        amp_dtype="bf16",
        apply_mask=True,
        mask_edges=True,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, (pred, img_path, cam_num) in enumerate(zip(preds, args.images, args.cam_nums)):
        pts3d = squeeze_batch(pred["pts3d"])           # (Hr, Wr, 3) world coords
        conf = squeeze_batch(pred.get("conf"))          # (Hr, Wr)
        mask = squeeze_batch(pred.get("mask"))          # (Hr, Wr, 1) or (Hr, Wr)
        intr = squeeze_batch(pred.get("intrinsics"))    # (3, 3)
        pose = squeeze_batch(pred.get("camera_poses"))  # (4, 4)
        mask = np.squeeze(mask)

        full_w, full_h = Image.open(img_path).size      # original resolution
        hr, wr = pts3d.shape[:2]

        np.savez_compressed(
            out_dir / f"camera_{cam_num}_recon.npz",
            pts3d=pts3d.astype(np.float32),
            conf=None if conf is None else conf.astype(np.float32),
            mask=mask.astype(bool),
            intrinsics=intr.astype(np.float64),
            camera_pose=pose.astype(np.float64),
            recon_hw=np.array([hr, wr]),
            full_wh=np.array([full_w, full_h]),
            cam_num=np.array([cam_num]),
            source_image=str(img_path),
        )
        print(f"  saved camera_{cam_num}_recon.npz  recon={wr}x{hr}  full={full_w}x{full_h}")

    print(f"Done -> {out_dir}")


if __name__ == "__main__":
    main()
