#!/usr/bin/env python3
"""Quick quality check for the walker pill-grasp dataset."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

root = Path(r"C:\Users\ryker\corgi-hackathon\datasets\walker_pill_grasp_test")
out = Path(r"C:\Users\ryker\corgi-hackathon\_cam_probe")
out.mkdir(exist_ok=True)

info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
print("=== DATASET SUMMARY ===")
print(f"episodes: {info['total_episodes']}")
print(f"frames:   {info['total_frames']}")
print(f"fps meta: {info['fps']}")
print(
    f"avg sec @meta fps: {info['total_frames'] / info['fps'] / max(info['total_episodes'], 1):.1f}s"
)
print(f"robot:    {info['robot_type']}")

ep = pq.read_table(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet").to_pandas()
print("\n=== EPISODES META ===")
print(ep.to_string())

data = pq.read_table(root / "data" / "chunk-000" / "file-000.parquet").to_pandas()
print("\n=== PER-EPISODE QUALITY ===")
for ei in sorted(data["episode_index"].unique()):
    d = data[data["episode_index"] == ei]
    n = len(d)
    print(f"\nepisode {int(ei)}: {n} frames (~{n / 11:.1f}s at ~11 Hz actual)")
    if "action" not in d.columns:
        print("  no action column")
        continue
    acts = np.stack(d["action"].to_list())
    deltas = np.abs(np.diff(acts, axis=0)).sum(axis=1) if n > 1 else np.array([0.0])
    print(f"  action shape: {acts.shape}")
    print(f"  mean |daction|/step: {float(deltas.mean()):.4f}")
    print(f"  max  |daction|/step: {float(deltas.max()):.4f}")
    print(f"  gripper {acts[0, -1]:.2f} -> {acts[-1, -1]:.2f}")
    print(
        f"  pan/lift start {acts[0, 0]:.1f}/{acts[0, 1]:.1f} "
        f"end {acts[-1, 0]:.1f}/{acts[-1, 1]:.1f}"
    )
    moving = float((deltas > 0.05).mean()) if len(deltas) else 0.0
    print(f"  fraction of steps with motion: {moving:.0%}")

vid = root / "videos" / "observation.images.wrist" / "chunk-000" / "file-000.mp4"
print(f"\nvideo: {vid} ({vid.stat().st_size / 1e6:.2f} MB)")
cap = cv2.VideoCapture(str(vid))
nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
print(f"video frames={nframes} fps={fps} duration={nframes / fps if fps else 0:.1f}s")
for pct, name in [(0.05, "ds_05"), (0.35, "ds_35"), (0.70, "ds_70"), (0.92, "ds_92")]:
    idx = min(nframes - 1, max(0, int(nframes * pct)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if ok:
        path = out / f"{name}.jpg"
        cv2.imwrite(str(path), frame)
        print(f"saved {path.name} (frame {idx})")
cap.release()
print("\nDONE")
