# Oliver Fritsche
# June 7, 2026
# CS 7180 Advanced Perception

"""Inspect a demo HDF5 and dump a few frames per camera — NO Isaac needed.

Prints the dataset structure (demo count, per-demo keys/shapes/dtypes, image
value ranges) and saves start/mid/end frames for each camera view so the
wrist/scene framing can be eyeballed across the approach->grasp->lift trajectory.

Run (pure h5py/PIL/numpy; the Isaac python has them, but any does):
    /isaac-sim/python.sh scripts/peek_demo.py --dir data/smoke --out /tmp/frames
"""
import argparse
import glob
import os

import h5py
import numpy as np
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("--file", default=None, help="HDF5 path (default: newest in --dir)")
p.add_argument("--dir", default="data/smoke", help="dir to search for the newest *.hdf5")
p.add_argument("--out", default="/tmp/frames", help="where to write extracted frames")
p.add_argument("--demos", type=int, default=1, help="how many demos to dump frames from")
a = p.parse_args()

path = a.file or sorted(glob.glob(os.path.join(a.dir, "*.hdf5")))[-1]
os.makedirs(a.out, exist_ok=True)
h = h5py.File(path, "r")

print("FILE:", path)
print("ATTRS:", {k: h.attrs[k] for k in h.attrs})
demos = list(h.keys())
print("NUM_DEMOS:", len(demos), "->", demos[: min(5, len(demos))])

g0 = h[demos[0]]
print("DEMO0_KEYS:")
for k in g0.keys():
    d = g0[k]
    info = f"{d.shape} {d.dtype}" if hasattr(d, "shape") else "(group)"
    extra = ""
    if hasattr(d, "shape") and d.ndim == 4:  # image stack (T,H,W,C)
        arr = d[:]
        extra = f"  min={arr.min()} max={arr.max()} mean={arr.mean():.1f}"
    print(f"  {k}: {info}{extra}")

cams = [k.replace("images_", "") for k in g0.keys() if k.startswith("images_")]
print("CAMERAS:", cams)

for di in range(min(a.demos, len(demos))):
    g = h[demos[di]]
    for cam in cams:
        key = f"images_{cam}"
        if key not in g:
            continue
        arr = g[key][:]
        T = arr.shape[0]
        for frac, nm in [(0.0, "start"), (0.5, "mid"), (0.95, "end")]:
            t = int(frac * (T - 1))
            out = os.path.join(a.out, f"d{di}_{cam}_{nm}.png")
            Image.fromarray(arr[t]).save(out)
            print("saved", out)
print("PEEK_DONE")
