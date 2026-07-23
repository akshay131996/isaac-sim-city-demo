"""Warehouse XL — a larger, busier warehouse demo. Isaac Sim 5.1, headless.

A NEW variant, separate from warehouse_forklift.py (which is left untouched):
  * Bigger room: 20 x 14 m (vs the original 10 x 10).
  * TWO vehicles: the forklift performs a box transfer between aisles, and a
    Leatherback car patrols a loop on the open floor.
  * More assets: multiple stocked shelving racks, pallets, cardboard boxes,
    KLT bins, plastic barrels, bottles, support pillars, and floor decals.

It reuses the proven KINEMATIC techniques from the original (waypoint-table
interpolation with smoothstep easing; the box "rides" the forks via a
pose-follow during a carry window; no physics stepping, only world.render()).
Two teaching additions over the original:
  * Multi-actor choreography — each vehicle has its own waypoint table, driven
    by the same interpolator. Scaling from 1 actor to N is data, not new code.
  * Auto-heading — the car FACES its direction of travel (yaw derived from its
    velocity each frame), so you never hand-tune per-waypoint yaw for it.

Env vars:
  SMOKE=1     render a few stills at key beats instead of the full video
  NUM_FRAMES  override frame count (default 600 = 30 s at 20 fps)

Run (on the pod):
  export OMNI_KIT_ACCEPT_EULA=yes
  xvfb-run -a -s "-screen 0 1280x720x24" \
      /workspace/isaac_env/bin/python3 -u warehouse_xl.py
  ffmpeg -framerate 20 -i /workspace/warehouse_xl_frames/frame_%04d.jpg \
         -c:v libx264 -pix_fmt yuv420p warehouse_xl.mp4
"""
import os
import numpy as np
import cv2

print("Starting Isaac Sim (headless)...")
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

from omni.isaac.core import World
from omni.isaac.core.objects import VisualCuboid
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.sensor import Camera
import omni
from pxr import Usd, UsdGeom, UsdLux, Gf

FRAMES_DIR = "/workspace/warehouse_xl_frames"
FPS = 20
NUM_FRAMES = int(os.environ.get("NUM_FRAMES", str(30 * FPS)))
SMOKE = os.environ.get("SMOKE") == "1"

# Room half-extents (room is 20 x 14 m). Walls kept low (5 m) so the elevated
# orbit camera always sees over them.
HX, HY, WALL_H = 10.0, 7.0, 5.0

world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()
assets_root = get_assets_root_path()
bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])

WH = "/Isaac/Environments/Simple_Warehouse/Props"


def report_bbox(prim_path, label):
    rng = bbox_cache.ComputeWorldBound(stage.GetPrimAtPath(prim_path)).ComputeAlignedRange()
    if rng.IsEmpty():
        print(f"BBOX {label}: EMPTY (asset failed to load!)")
        return
    s = rng.GetMax() - rng.GetMin()
    print(f"BBOX {label}: size=({s[0]:.2f},{s[1]:.2f},{s[2]:.2f})")


def place(usd_path, prim_path, pos, yaw_deg=0.0, scale=None):
    add_reference_to_stage(usd_path=assets_root + usd_path, prim_path=prim_path)
    xf = XFormPrim(prim_path)
    xf.set_world_pose(position=np.array(pos, dtype=float),
                      orientation=euler_angles_to_quat(np.array([0.0, 0.0, np.radians(yaw_deg)])))
    if scale is not None:
        xf.set_local_scale(np.array([scale] * 3, dtype=float))
    return xf


# --------------------------------------------------------------- the room --
VisualCuboid(prim_path="/World/Floor", position=np.array([0, 0, -0.05]),
             scale=np.array([2 * HX, 2 * HY, 0.1]), color=np.array([0.42, 0.42, 0.46]))
for name, c, s in [
    ("WallN", [0,  HY, WALL_H / 2], [2 * HX, 0.2, WALL_H]),
    ("WallS", [0, -HY, WALL_H / 2], [2 * HX, 0.2, WALL_H]),
    ("WallE", [ HX, 0, WALL_H / 2], [0.2, 2 * HY, WALL_H]),
    ("WallW", [-HX, 0, WALL_H / 2], [0.2, 2 * HY, WALL_H]),
]:
    VisualCuboid(prim_path=f"/World/{name}", position=np.array(c, dtype=float),
                 scale=np.array(s, dtype=float), color=np.array([0.78, 0.76, 0.71]))

dome = UsdLux.DomeLight.Define(stage, "/World/Dome")
dome.CreateIntensityAttr(1400)
sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
sun.CreateIntensityAttr(1600)
UsdGeom.XformCommonAPI(sun.GetPrim()).SetRotate(Gf.Vec3f(-55, 0, 30))

# --------------------------------------------------- shelving racks (many) --
ORANGE = np.array([0.85, 0.45, 0.1])
_rack_n = 0


def build_rack(cx, cy, yaw_deg):
    """A 2.4 m-wide, 2-tier pallet rack: 4 uprights + 2 shelf slabs, stocked
    with boxes/bins. Purely visual cuboids + prop refs (no physics here)."""
    global _rack_n
    _rack_n += 1
    root = f"/World/Racks/R{_rack_n}"
    yr = np.radians(yaw_deg)

    def tw(lx, ly, z):  # rack-local -> world
        return np.array([cx + lx * np.cos(yr) - ly * np.sin(yr),
                         cy + lx * np.sin(yr) + ly * np.cos(yr), z])

    for i, (lx, ly) in enumerate([(-1.2, -0.5), (-1.2, 0.5), (1.2, -0.5), (1.2, 0.5)]):
        VisualCuboid(prim_path=f"{root}/Post{i}", position=tw(lx, ly, 1.1),
                     scale=np.array([0.09, 0.09, 2.2]), color=ORANGE)
    for j, z in enumerate([0.75, 1.6]):
        VisualCuboid(prim_path=f"{root}/Shelf{j}", position=tw(0, 0, z),
                     orientation=euler_angles_to_quat(np.array([0, 0, yr])),
                     scale=np.array([2.5, 1.1, 0.06]), color=np.array([0.33, 0.33, 0.36]))
    # Light stock (2 physics-carrying props/rack). The rack frame itself is
    # cheap VisualCuboids (no collider). PhysX crashed on ~3x this asset
    # budget, so we keep the collision-mesh count near the proven-safe level.
    place(f"{WH}/SM_CardBoxA_01.usd", f"{root}/S0", tw(-0.5, 0, 0.80), yaw_deg + 10)
    place("/Isaac/Props/KLT_Bin/small_KLT.usd", f"{root}/S1", tw(0.5, 0, 0.80), yaw_deg)


# racks line the north and south walls (facing inward). 6 racks fill the
# larger room; the frames are cheap cuboids, only the 2 props/rack cost PhysX.
for cx in (-6.0, 0.0, 6.0):
    build_rack(cx, HY - 1.3, 0)        # north wall
    build_rack(cx, -HY + 1.3, 0)       # south wall

# ----------------------------------------------------- scattered dressing --
# (Dropped the 9 m pillars and bottle clusters that pushed PhysX over its
# limit; kept barrels + a staged pallet for a busy-but-stable scene.)
place(f"{WH}/SM_BarelPlastic_A_01.usd", "/World/Barrel1", [-8.6, -4.5, 0.0])
place(f"{WH}/SM_BarelPlastic_A_02.usd", "/World/Barrel2", [-8.1, -4.7, 0.0])
place(f"{WH}/SM_BarelPlastic_A_01.usd", "/World/Barrel3", [8.4, 4.6, 0.0], 40)
place("/Isaac/Props/Pallet/pallet.usd", "/World/Pallet1", [7.5, -5.0, 0.0], 20)
place(f"{WH}/SM_CardBoxA_01.usd", "/World/PBox1", [7.5, -5.0, 0.15], 20)
place(f"{WH}/SM_FloorDecal_Keepclear.usd", "/World/Decal1", [0.0, 0.0, 0.02])
place(f"{WH}/SM_FloorDecal_Keepclear.usd", "/World/Decal2", [0.0, -3.5, 0.02], 90)

# ----------------------------------------------------- hero box + forklift --
BOX_START = np.array([-8.4, 1.5, 0.0])
hero = place(f"{WH}/SM_CardBoxA_01.usd", "/World/HeroBox", BOX_START)
place("/Isaac/Props/Forklift/forklift.usd", "/World/Forklift", [0.0, -3.0, 0.0], 0)
forklift = XFormPrim("/World/Forklift")

# second vehicle: the CAR (Leatherback). cm-authored asset -> scale 0.05 gives
# a ~2.1 m car.
car = place("/Isaac/Robots/NVIDIA/Leatherback/leatherback.usd", "/World/Car",
            [6.0, 0.0, 0.05], scale=0.05)

report_bbox("/World/Forklift", "forklift")
report_bbox("/World/Car", "car")
report_bbox("/World/HeroBox", "hero_box")
report_bbox("/World/Racks/R1", "rack")

fork_prim = None
for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/Forklift")):
    if prim.GetPath().pathString == "/World/Forklift":
        continue
    if prim.GetName().lower().endswith("fork") and prim.IsA(UsdGeom.Xformable):
        fork_prim = XFormPrim(prim.GetPath().pathString)
        print("FORK PRIM:", prim.GetPath().pathString)
        break

# ------------------------------------------------------------ choreography --
# Forklift: (t, x, y, yaw_deg, fork_h). Fork axis is the asset's local -y, so
# yaw -90 points the forks toward -x (aisle to the west), yaw +90 toward +x.
# Carry: while active, the box rides FORK_FORWARD metres along the fork axis.
FORK_WP = [
    (0.0,  0.0, -3.0,    0, 0.05),   # rest, mid-floor
    (2.0,  0.0, -3.0,    0, 0.05),
    (5.0, -5.5,  1.5,  -90, 0.05),   # drive to the west aisle, square up
    (7.5, -6.9,  1.5,  -90, 0.05),   # creep in, forks under the box
    (10.0, -6.9, 1.5,  -90, 0.90),   # lift
    (12.5, -5.0, 1.5,  -90, 0.90),   # back out with the load
    (16.0,  0.0, -1.0,  90, 0.90),   # cross the floor, turning to face +x
    (19.0,  5.5, -1.5,  90, 0.90),   # drive to the east aisle
    (21.5,  6.9, -1.5,  90, 0.90),   # creep in over the drop pallet
    (23.5,  6.9, -1.5,  90, 0.15),   # lower the box
    (26.0,  5.0, -1.5,  90, 0.05),   # back away
    (30.0,  0.0, -4.0, 180, 0.05),   # park
]
CARRY_START, CARRY_END = 8.0, 23.5
FORK_FORWARD = 1.5

# Car: POSITION-only waypoints; heading is derived from travel direction each
# frame (auto-heading), so no per-waypoint yaw tuning. A patrol loop around
# the open eastern/southern floor, clear of the forklift's aisle work.
CAR_WP = [
    (0.0,  6.0,  0.0), (4.0,  8.0,  4.5), (8.0,  3.0,  5.5),
    (12.0, -3.0, 5.5), (15.0, -6.0, 3.0), (19.0, -6.0, -3.0),
    (23.0, 3.0, -3.5), (27.0, 8.0, -3.0), (30.0, 8.0,  1.0),
]
CAR_YAW_OFFSET = 0.0   # tune from smoke test if the car faces wrong


def smoothstep(u):
    return u * u * (3 - 2 * u)


def interp4(wp, t):
    if t <= wp[0][0]:
        return wp[0][1:]
    for a, b in zip(wp, wp[1:]):
        if a[0] <= t <= b[0]:
            u = smoothstep((t - a[0]) / (b[0] - a[0]))
            return tuple(a[i] + u * (b[i] - a[i]) for i in range(1, len(a)))
    return wp[-1][1:]


def car_pose(t):
    """Position from the table; yaw from the direction to a point just ahead."""
    x, y = interp4(CAR_WP, t)
    xa, ya = interp4(CAR_WP, min(t + 0.25, CAR_WP[-1][0]))
    yaw = np.arctan2(ya - y, xa - x) if (xa - x or ya - y) else 0.0
    return x, y, yaw + np.radians(CAR_YAW_OFFSET)


def apply_frame(t):
    fx, fy, fyaw, fh = interp4(FORK_WP, t)
    fyr = np.radians(fyaw)
    forklift.set_world_pose(position=np.array([fx, fy, 0.0]),
                            orientation=euler_angles_to_quat(np.array([0, 0, fyr])))
    if fork_prim is not None:
        p, q = fork_prim.get_local_pose()
        fork_prim.set_local_pose(translation=np.array([p[0], p[1], fh]), orientation=q)
    if CARRY_START <= t <= CARRY_END:
        bx = fx + FORK_FORWARD * np.sin(fyr)
        by = fy - FORK_FORWARD * np.cos(fyr)
        hero.set_world_pose(position=np.array([bx, by, fh]),
                            orientation=euler_angles_to_quat(np.array([0, 0, fyr])))
    cx, cy, cyaw = car_pose(t)
    car.set_world_pose(position=np.array([cx, cy, 0.05]),
                       orientation=euler_angles_to_quat(np.array([0, 0, cyaw])))


# ---------------------------------------------------------------- camera ----
camera = Camera(prim_path="/World/Cam", position=np.array([0, 0, 12]),
                frequency=FPS, resolution=(1280, 720))
camera.initialize()
world.reset()
camera.initialize()
usd_cam = UsdGeom.Camera(stage.GetPrimAtPath("/World/Cam"))
usd_cam.GetFocalLengthAttr().Set(usd_cam.GetFocalLengthAttr().Get() * 0.42)  # wide
usd_cam.GetFStopAttr().Set(0.0)

CENTER = np.array([0.0, 0.0, 0.8])


def place_camera(t):
    """Slow elevated orbit, high enough (z=11) to clear the 5 m walls."""
    theta = np.radians(215) + 2 * np.pi * (t / (NUM_FRAMES / FPS)) * 0.5
    pos = CENTER + np.array([13.5 * np.cos(theta), 10.0 * np.sin(theta), 11.0])
    d = CENTER - pos
    yaw = np.arctan2(d[1], d[0])
    pitch = np.arctan2(-d[2], np.linalg.norm(d[:2]))
    camera.set_world_pose(position=pos, orientation=euler_angles_to_quat(np.array([0.0, pitch, yaw])))


os.makedirs(FRAMES_DIR, exist_ok=True)
for f in os.listdir(FRAMES_DIR):
    os.remove(os.path.join(FRAMES_DIR, f))

print("Warming up renderer...")
apply_frame(0.0)
place_camera(0.0)
for _ in range(40):
    world.render()


def capture(path):
    img = camera.get_rgba()
    if img is not None and img.size > 0:
        cv2.imwrite(path, cv2.cvtColor(img[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR))
        return True
    return False


if SMOKE:
    for t in [0.0, 7.5, 10.0, 16.0, 21.5, 27.0]:
        apply_frame(t)
        place_camera(t)
        for _ in range(15):
            world.render()
        fx, fy, fyaw, fh = interp4(FORK_WP, t)
        cx, cy, _ = car_pose(t)
        ok = capture(f"/workspace/smoke_xl_t{int(t * 10):03d}.jpg")
        print(f"SMOKE t={t}: {'saved' if ok else 'EMPTY'} "
              f"fork=({fx:.1f},{fy:.1f},yaw{fyaw:.0f},h{fh:.2f}) car=({cx:.1f},{cy:.1f})")
else:
    saved = 0
    for i in range(NUM_FRAMES):
        t = i / FPS
        apply_frame(t)
        place_camera(t)
        world.render()
        if capture(os.path.join(FRAMES_DIR, f"frame_{i:04d}.jpg")):
            saved += 1
        if (i + 1) % 100 == 0:
            print(f"Recorded {i + 1}/{NUM_FRAMES} frames")
    print(f"Done. Saved {saved}/{NUM_FRAMES} frames to {FRAMES_DIR}")

simulation_app.close()
