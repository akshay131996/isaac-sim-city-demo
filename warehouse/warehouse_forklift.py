"""Warehouse forklift demo — Isaac Sim 5.1, headless.

A 10 x 10 x 10 m warehouse room built from primitives, two aisles of pallets
stacked with crates and cardboard boxes, and a forklift that carries one box
from aisle A to aisle B.

HOW THE ANIMATION WORKS (the core idea to learn):
The forklift is animated *kinematically* — we set its pose every frame from a
waypoint table, we never step the physics engine. This is how cinematic /
synthetic-data shots are usually made: deterministic, stable, no controller
tuning. The box "rides" the forklift because, while the carry flag is on, we
copy the forklift's pose (plus a forward/height offset) onto the box each
frame — pose-following instead of USD reparenting.

Env vars:
  SMOKE=1     render a handful of stills at key waypoints instead of the video
  NUM_FRAMES  override frame count (default 480 = 24 s at 20 fps)
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

FRAMES_DIR = "/workspace/warehouse_frames"
FPS = 20
DURATION_S = 24.0
NUM_FRAMES = int(os.environ.get("NUM_FRAMES", str(int(DURATION_S * FPS))))
SMOKE = os.environ.get("SMOKE") == "1"
WARMUP_STEPS = 40

world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()
assets_root = get_assets_root_path()
bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])


def report_bbox(prim_path, label):
    """Print an asset's world bounding box. ALWAYS do this after loading an
    asset — it catches missing assets (empty bbox) and unit mismatches (a
    'car' that is 80 m long was authored in centimeters)."""
    rng = bbox_cache.ComputeWorldBound(stage.GetPrimAtPath(prim_path)).ComputeAlignedRange()
    if rng.IsEmpty():
        print(f"BBOX {label}: EMPTY (asset failed to load!)")
        return None
    size = rng.GetMax() - rng.GetMin()
    print(f"BBOX {label}: size=({size[0]:.2f}, {size[1]:.2f}, {size[2]:.2f}) "
          f"min_z={rng.GetMin()[2]:.2f}")
    return rng


def place(usd_path, prim_path, pos, yaw_deg=0.0, scale=None):
    """Reference an asset into the stage and pose it. XFormPrim handles any
    existing transform stack on the asset (XformCommonAPI silently fails on
    incompatible ones — hard-won lesson)."""
    add_reference_to_stage(usd_path=assets_root + usd_path, prim_path=prim_path)
    xf = XFormPrim(prim_path)
    xf.set_world_pose(position=np.array(pos, dtype=float),
                      orientation=euler_angles_to_quat(np.array([0.0, 0.0, np.radians(yaw_deg)])))
    if scale is not None:
        xf.set_local_scale(np.array([scale] * 3, dtype=float))
    return xf


# ---------------------------------------------------------------- the room --
# 10 x 10 m floor, four 10 m tall walls. Plain colored cuboids; no ceiling so
# the dome light can illuminate the interior.
ROOM = 10.0
VisualCuboid(prim_path="/World/Floor", position=np.array([0, 0, -0.05]),
             scale=np.array([ROOM, ROOM, 0.1]), color=np.array([0.45, 0.45, 0.48]))
wall_specs = [  # (name, center, scale)
    ("WallN", [0,  ROOM / 2, 5], [ROOM, 0.15, ROOM]),
    ("WallS", [0, -ROOM / 2, 5], [ROOM, 0.15, ROOM]),
    ("WallE", [ ROOM / 2, 0, 5], [0.15, ROOM, ROOM]),
    ("WallW", [-ROOM / 2, 0, 5], [0.15, ROOM, ROOM]),
]
for name, c, s in wall_specs:
    VisualCuboid(prim_path=f"/World/{name}", position=np.array(c, dtype=float),
                 scale=np.array(s, dtype=float), color=np.array([0.75, 0.73, 0.68]))

# -------------------------------------------------------------- lighting ----
dome = UsdLux.DomeLight.Define(stage, "/World/Dome")
dome.CreateIntensityAttr(1500)
sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
sun.CreateIntensityAttr(1500)
UsdGeom.XformCommonAPI(sun.GetPrim()).SetRotate(Gf.Vec3f(-55, 0, 25))

# ---------------------------------------------------------------- aisles ----
# Two rows of pallets against the E and W walls. Each pallet gets a random-ish
# load of KLT crates or cardboard boxes; one pallet in aisle B stays empty as
# the drop-off target.
PALLET = "/Isaac/Props/Pallet/pallet.usd"
KLT = "/Isaac/Props/KLT_Bin/small_KLT.usd"
CARDBOX = "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxA_01.usd"

aisle_x = {"A": -3.9, "B": 3.9}
pallet_ys = [-3.0, -1.0, 1.0, 3.0]
n = 0
for aisle, ax in aisle_x.items():
    for y in pallet_ys:
        n += 1
        place(PALLET, f"/World/Aisles/Pallet_{n}", [ax, y, 0.0], yaw_deg=90)
        if aisle == "B" and y == 1.0:
            continue  # drop-off pallet stays empty
        # alternate crate/box loads for visual variety
        if n % 2 == 0:
            place(KLT, f"/World/Aisles/Load_{n}a", [ax, y - 0.2, 0.30])
            place(KLT, f"/World/Aisles/Load_{n}b", [ax, y + 0.2, 0.30])
        else:
            place(CARDBOX, f"/World/Aisles/Load_{n}a", [ax, y, 0.30], yaw_deg=15)

# hero box: sits on the floor at the front of aisle A; the forklift moves it
BOX_START = np.array([-2.9, -1.0, 0.0])
hero = place(CARDBOX, "/World/HeroBox", BOX_START)

# ---------------------------------------------------------------- forklift --
forklift = place("/Isaac/Props/Forklift/forklift.usd", "/World/Forklift", [0.0, 3.5, 0.0], yaw_deg=-90)

report_bbox("/World/Forklift", "forklift")
report_bbox("/World/HeroBox", "hero_box")
report_bbox("/World/Aisles/Pallet_1", "pallet")

# Try to find the fork carriage inside the forklift asset so the forks
# visibly rise. If the asset doesn't expose one, the box still animates.
fork_prim = None
for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/Forklift")):
    # skip the root prim itself — its name ("Forklift") also contains "fork"
    if prim.GetPath().pathString == "/World/Forklift":
        continue
    # every part is named S_Forklift... so "fork" alone matches everything;
    # the fork carriage part is S_ForkliftFork
    if prim.GetName().lower().endswith("fork") and prim.IsA(UsdGeom.Xformable):
        fork_prim = XFormPrim(prim.GetPath().pathString)
        print("FORK PRIM:", prim.GetPath().pathString)
        break
if fork_prim is None:
    print("FORK PRIM: none found (box will animate without fork motion)")

# ------------------------------------------------------------ choreography --
# Waypoint table: (time_s, x, y, yaw_deg, fork_height_m). Yaw values are kept
# "unwrapped" (continuous) so interpolation turns the intended direction.
#
# IMPORTANT (found via smoke test): this forklift asset's forks point along
# its local -y axis, so fork_direction_angle = yaw - 90. yaw=0 drives forks
# down the corridor (-y), yaw=-90 points them at aisle A (-x), yaw=+90 at
# aisle B (+x). Never assume an asset's forward axis — check a render.
WAYPOINTS = [
    (0.0,  0.0,  3.5,    0, 0.05),   # start mid-corridor, forks toward -y
    (3.0, -0.3, -1.0,    0, 0.05),   # drive down the corridor
    (5.0, -0.3, -1.0,  -90, 0.05),   # turn in place: forks at aisle A (-x)
    (7.0, -1.4, -1.0,  -90, 0.05),   # creep in: forks slide under the box
    (9.0, -1.4, -1.0,  -90, 0.90),   # lift
    (11.0, -0.3, -1.0,  -90, 0.90),  # back out with the load
    (14.0,  0.0,  1.0,   90, 0.90),  # 180-degree turn, line up on aisle B
    (16.0,  1.7,  1.0,   90, 0.90),  # creep toward the empty pallet
    (18.0,  1.7,  1.0,   90, 0.90),  # pause
    (20.0,  1.7,  1.0,   90, 0.15),  # lower the box to the floor
    (22.0,  0.5,  1.0,   90, 0.05),  # back away
    (24.0,  0.2,  1.0,   90, 0.05),  # rest
]
CARRY_START, CARRY_END = 7.5, 20.0   # box follows the forks in this window
FORK_FORWARD = 1.5                   # box offset along the fork axis (m)


def smoothstep(u):
    """Ease-in/ease-out; raw linear interpolation looks robotic."""
    return u * u * (3 - 2 * u)


def pose_at(t):
    """Interpolate the waypoint table at time t."""
    if t <= WAYPOINTS[0][0]:
        _, x, y, yaw, fh = WAYPOINTS[0]
        return x, y, yaw, fh
    for (t0, x0, y0, w0, f0), (t1, x1, y1, w1, f1) in zip(WAYPOINTS, WAYPOINTS[1:]):
        if t0 <= t <= t1:
            u = smoothstep((t - t0) / (t1 - t0))
            return (x0 + u * (x1 - x0), y0 + u * (y1 - y0),
                    w0 + u * (w1 - w0), f0 + u * (f1 - f0))
    _, x, y, yaw, fh = WAYPOINTS[-1]
    return x, y, yaw, fh


def apply_frame(t):
    x, y, yaw, fh = pose_at(t)
    yaw_r = np.radians(yaw)
    forklift.set_world_pose(position=np.array([x, y, 0.0]),
                            orientation=euler_angles_to_quat(np.array([0.0, 0.0, yaw_r])))
    if fork_prim is not None:
        # raise the fork carriage locally within the forklift
        p, q = fork_prim.get_local_pose()
        fork_prim.set_local_pose(translation=np.array([p[0], p[1], fh]), orientation=q)
    if CARRY_START <= t <= CARRY_END:
        # the pose-follow "attach": box = forklift pose + offset along the
        # fork axis (local -y => world (sin(yaw), -cos(yaw))) + fork height
        bx = x + FORK_FORWARD * np.sin(yaw_r)
        by = y - FORK_FORWARD * np.cos(yaw_r)
        hero.set_world_pose(position=np.array([bx, by, fh]),
                            orientation=euler_angles_to_quat(np.array([0.0, 0.0, yaw_r])))
    # before CARRY_START the box sits at BOX_START; after CARRY_END it keeps
    # the last pose we gave it — nothing to do in either case.


# ---------------------------------------------------------------- camera ----
camera = Camera(prim_path="/World/Cam", position=np.array([4.4, -4.4, 5.5]),
                frequency=FPS, resolution=(1280, 720))
camera.initialize()
world.reset()
camera.initialize()

# The Isaac Camera's default gives a narrow (~24 deg) FOV — fine for
# close-ups, useless for an overview shot. Widen to ~55 deg and disable depth
# of field. We write the USD camera attributes directly: the Camera wrapper's
# set_focal_length/set_f_stop segfaulted in this build.
usd_cam = UsdGeom.Camera(stage.GetPrimAtPath("/World/Cam"))
fl = usd_cam.GetFocalLengthAttr().Get()
print(f"CAM usd focal={fl} aperture={usd_cam.GetHorizontalApertureAttr().Get()}")
usd_cam.GetFocalLengthAttr().Set(fl * 0.4)   # ~2.5x wider FOV
usd_cam.GetFStopAttr().Set(0.0)
print(f"CAM new focal={usd_cam.GetFocalLengthAttr().Get()}")

CAM_LOOK = np.array([0.0, 0.3, 0.6])
CAM_POS = np.array([4.4, -4.4, 5.5])
d = CAM_LOOK - CAM_POS
cam_yaw = np.arctan2(d[1], d[0])
cam_pitch = np.arctan2(-d[2], np.linalg.norm(d[:2]))
camera.set_world_pose(position=CAM_POS,
                      orientation=euler_angles_to_quat(np.array([0.0, cam_pitch, cam_yaw])))

os.makedirs(FRAMES_DIR, exist_ok=True)
for f in os.listdir(FRAMES_DIR):
    os.remove(os.path.join(FRAMES_DIR, f))

print("Warming up renderer...")
apply_frame(0.0)
for _ in range(WARMUP_STEPS):
    world.render()


def capture(path):
    img = camera.get_rgba()
    if img is not None and img.size > 0:
        cv2.imwrite(path, cv2.cvtColor(img[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR))
        return True
    return False


if SMOKE:
    # stills at the key beats of the choreography
    for t in [0.0, 6.5, 9.0, 14.0, 19.0, 23.0]:
        apply_frame(t)
        for _ in range(15):
            world.render()
        ok = capture(f"/workspace/smoke_t{int(t*10):03d}.jpg")
        print(f"SMOKE t={t}: {'saved' if ok else 'EMPTY'}")
else:
    saved = 0
    for i in range(NUM_FRAMES):
        apply_frame(i / FPS)
        world.render()
        if capture(os.path.join(FRAMES_DIR, f"frame_{i:04d}.jpg")):
            saved += 1
        if (i + 1) % 50 == 0:
            print(f"Recorded {i + 1}/{NUM_FRAMES} frames")
    print(f"Done. Saved {saved}/{NUM_FRAMES} frames to {FRAMES_DIR}")

simulation_app.close()
