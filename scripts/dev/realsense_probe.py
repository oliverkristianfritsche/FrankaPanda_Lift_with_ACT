"""Probe this Isaac Sim install for Intel RealSense camera assets + their camera prims,
so we can attach a hardware-accurate wrist camera (see make_wrist_camera).

Prints the assets root, lists the RealSense sensor dir, and dumps the prim tree
(highlighting Camera prims + their focal/aperture) of the first RealSense usd found.
"""
import argparse
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
a.headless = True
a.enable_cameras = True
app = AppLauncher(a).app

import omni.client  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402

try:
    from isaacsim.storage.native import get_assets_root_path
except Exception:  # noqa: BLE001
    from omni.isaac.nucleus import get_assets_root_path

root = get_assets_root_path()
print("ASSETS_ROOT", root, flush=True)


def listdir(url):
    res, entries = omni.client.list(url)
    print(f"LIST {url} -> {res}", flush=True)
    for e in entries:
        print("   ", e.relative_path, flush=True)
    return [e.relative_path for e in entries]


candidates = []
for sub in ["/Isaac/Sensors/Intel/RealSense/", "/Isaac/Sensors/Intel/", "/Isaac/Sensors/"]:
    try:
        names = listdir(root + sub)
        for n in names:
            if n.lower().endswith(".usd") and ("rsd" in n.lower() or "realsense" in n.lower() or "d4" in n.lower()):
                candidates.append(root + sub + n)
    except Exception as e:  # noqa: BLE001
        print("ERR_LIST", sub, repr(e), flush=True)

# Also try common direct paths.
for direct in ["/Isaac/Sensors/Intel/RealSense/rsd455.usd",
               "/Isaac/Sensors/Intel/RealSense/RSD455/rsd455.usd"]:
    candidates.append(root + direct)

seen = set()
for url in candidates:
    if url in seen:
        continue
    seen.add(url)
    res, _ = omni.client.stat(url)
    ok = "OK" in str(res)
    print(f"STAT {url} -> {res} ok={ok}", flush=True)
    if not ok:
        continue
    st = Usd.Stage.Open(url)
    if not st:
        print("   could not open", flush=True)
        continue
    print("   DEFAULT_PRIM", st.GetDefaultPrim().GetPath() if st.GetDefaultPrim() else None, flush=True)
    for prim in st.Traverse():
        if prim.IsA(UsdGeom.Camera):
            cam = UsdGeom.Camera(prim)
            fl = cam.GetFocalLengthAttr().Get()
            ha = cam.GetHorizontalApertureAttr().Get()
            print(f"   CAMERA {prim.GetPath()}  focal={fl} h_aperture={ha}", flush=True)
    break  # first openable asset is enough

print("PROBE_DONE", flush=True)
app.close()
