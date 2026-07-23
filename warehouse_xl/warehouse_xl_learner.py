"""Warehouse XL (LEARNER EDITION) — a heavily-annotated walkthrough.

This is a teaching copy of warehouse_xl.py. The code is functionally identical
and still runs, but every function has a full docstring and the non-obvious
lines are explained inline. Read it top to bottom like a tutorial.

=============================================================================
BIG PICTURE: what this program is
=============================================================================
It builds a virtual warehouse in NVIDIA Isaac Sim, animates a forklift moving
a box between two aisles while a car drives around, and saves the rendered
camera view to image files that ffmpeg later stitches into a video.

Three ideas underpin the whole thing:

1. USD (Universal Scene Description) is the scene database. Everything in the
   world — the floor, a wall, the forklift, a light, the camera — is a "prim"
   (a node) living at a path like "/World/Forklift". You build a scene by
   creating prims and/or by *referencing* prebuilt asset files (.usd) into
   paths on your stage. Each prim can carry a transform (position/rotation/
   scale), geometry, materials, physics, etc.

2. This is a KINEMATIC animation, not a physics simulation. We never ask the
   physics engine to advance time (no world.step()); we just *set* each
   object's pose every frame from a script and then render a picture
   (world.render()). It's deterministic and simple — the price is that we,
   not physics, are responsible for making motion look believable.

3. Everything animated is driven by WAYPOINT TABLES + INTERPOLATION. A table
   lists a few (time, pose) keyframes; each frame we look up the current time,
   find the surrounding keyframes, and blend between them. One tiny
   interpolator drives the forklift, the fork height, and the car.

=============================================================================
WHAT'S IN THE SCENE
=============================================================================
  * A 20 x 14 m room (floor + 4 walls), lit by a sun + dome light.
  * Six 2-tier shelving racks (built from boxes/cuboids) stocked with cardboard
    boxes and KLT bins, plus barrels, a staged pallet, and floor decals.
  * A forklift that drives to the west aisle, slides its forks under a box,
    lifts it, carries it across, and sets it down at the east aisle.
  * A car (NVIDIA Leatherback) patrolling a loop on the open floor.
  * One orbiting camera that films the whole thing.

Env vars:
  SMOKE=1     render a few stills at key beats instead of the full video
  NUM_FRAMES  override frame count (default 600 = 30 s at 20 fps)

Run (on the pod):
  export OMNI_KIT_ACCEPT_EULA=yes
  xvfb-run -a -s "-screen 0 1280x720x24" \
      /workspace/isaac_env/bin/python3 -u warehouse_xl_learner.py
  ffmpeg -framerate 20 -i /workspace/warehouse_xl_frames/frame_%04d.jpg \
         -c:v libx264 -pix_fmt yuv420p warehouse_xl.mp4
"""
import os
import numpy as np   # all vector/rotation math; poses are numpy arrays
import cv2           # OpenCV: converts the rendered pixels and writes JPEGs

# ---------------------------------------------------------------------------
# BOOT ISAAC SIM. This MUST happen before importing anything from omni.* /
# pxr, because SimulationApp starts the underlying "Kit" application and loads
# the extensions those modules live in. Import them earlier and you get import
# errors. headless=True means "no on-screen window" (we render to files).
# ---------------------------------------------------------------------------
print("Starting Isaac Sim (headless)...")
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

# Now that Kit is running, these imports resolve:
from omni.isaac.core import World                    # owns the stage + sim loop
from omni.isaac.core.objects import VisualCuboid     # a colored box prim (no physics)
from omni.isaac.core.prims import XFormPrim          # a handle to move any prim
from omni.isaac.core.utils.nucleus import get_assets_root_path  # base URL for NVIDIA assets
from omni.isaac.core.utils.stage import add_reference_to_stage  # pull a .usd into the scene
from omni.isaac.core.utils.rotations import euler_angles_to_quat  # (roll,pitch,yaw)->quaternion
from omni.isaac.sensor import Camera                 # a camera sensor we can read pixels from
import omni                                           # for omni.usd.get_context()
from pxr import Usd, UsdGeom, UsdLux, Gf              # raw USD: prims, geometry, lights, math types

# ---- configuration constants ----------------------------------------------
FRAMES_DIR = "/workspace/warehouse_xl_frames"        # where rendered JPEGs go
FPS = 20                                             # frames per second of the output video
# NUM_FRAMES: 600 by default (= 30 s * 20 fps). os.environ.get lets you override
# it from the shell (NUM_FRAMES=100 python ...) without editing the file.
NUM_FRAMES = int(os.environ.get("NUM_FRAMES", str(30 * FPS)))
# SMOKE mode renders a handful of test stills instead of the full video — the
# cheap "did I break anything?" check you run before a long render.
SMOKE = os.environ.get("SMOKE") == "1"

# Room half-extents. The room spans 20 m in x and 14 m in y, so the half-widths
# are 10 and 7; walls are 5 m tall (kept low so the high camera sees over them).
HX, HY, WALL_H = 10.0, 7.0, 5.0

# ---- core Isaac objects ----------------------------------------------------
# World owns the USD stage, the physics scene, and the step/render loop.
# stage_units_in_meters=1.0 means "1 world unit == 1 metre" (so all our numbers
# are metres). Getting this wrong is a classic bug (assets authored in cm look
# 100x too big).
world = World(stage_units_in_meters=1.0)
# The stage is the live USD scene graph. We grab it to create prims directly.
stage = omni.usd.get_context().get_stage()
# NVIDIA hosts the warehouse/robot assets on a CDN; this returns that base URL,
# to which we append asset sub-paths.
assets_root = get_assets_root_path()
# A BBoxCache computes world-space bounding boxes of prims (used by report_bbox
# below to sanity-check that assets loaded at a sane size).
bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])

# Short alias for the Simple_Warehouse prop folder so paths below stay short.
WH = "/Isaac/Environments/Simple_Warehouse/Props"


def report_bbox(prim_path, label):
    """Print the world-space size of a prim's bounding box.

    WHY THIS EXISTS: `add_reference_to_stage` does NOT raise if the asset URL
    is wrong — it silently leaves an empty prim. And assets authored in the
    wrong unit (cm vs m) load 100x too big/small. One bbox print catches both:
    an EMPTY box means the asset didn't load; a wildly wrong size means a unit
    or scale problem. Always sanity-check assets this way after loading them.

    Args:
        prim_path: the "/World/..." path of the prim to measure.
        label:     a human-readable name to print alongside the size.
    """
    # ComputeWorldBound walks the prim's subtree and returns an oriented box;
    # ComputeAlignedRange converts it to an axis-aligned min/max range.
    rng = bbox_cache.ComputeWorldBound(stage.GetPrimAtPath(prim_path)).ComputeAlignedRange()
    if rng.IsEmpty():
        print(f"BBOX {label}: EMPTY (asset failed to load!)")
        return
    s = rng.GetMax() - rng.GetMin()   # (max - min) = the box's size in each axis
    print(f"BBOX {label}: size=({s[0]:.2f},{s[1]:.2f},{s[2]:.2f})")


def place(usd_path, prim_path, pos, yaw_deg=0.0, scale=None):
    """Reference an asset file into the scene and position it. Returns a handle.

    This is the single most-used helper: it turns a one-line call into
    "load this .usd, put it at this spot, facing this way, at this size".

    Args:
        usd_path:  asset sub-path appended to assets_root, e.g.
                   "/Isaac/Props/Forklift/forklift.usd".
        prim_path: where to mount it on the stage, e.g. "/World/Forklift".
                   Must be unique — reuse a path and you overwrite the prim.
        pos:       [x, y, z] world position in metres.
        yaw_deg:   rotation about the vertical (z) axis, in DEGREES. Yaw is the
                   only rotation a floor vehicle/prop needs (which way it faces).
        scale:     optional uniform scale factor. Used for the car, whose asset
                   is authored in centimetres, so scale=0.05 shrinks it to a
                   real ~2 m car.

    Returns:
        An XFormPrim handle you can call .set_world_pose(...) on later to
        animate the prim.
    """
    # add_reference_to_stage creates prim_path and "references" the external
    # .usd into it (like an include/import). The asset's contents now live under
    # that path.
    add_reference_to_stage(usd_path=assets_root + usd_path, prim_path=prim_path)
    # XFormPrim is a thin wrapper that lets us set/get a prim's transform.
    xf = XFormPrim(prim_path)
    # set_world_pose takes a position and an ORIENTATION AS A QUATERNION, not
    # Euler angles. A quaternion is a 4-number (w,x,y,z) way to store a 3D
    # rotation with no gimbal-lock. We build ours from Euler angles
    # [roll, pitch, yaw]; here roll=pitch=0 and only yaw is nonzero. np.radians
    # converts degrees -> radians (the math functions all want radians).
    xf.set_world_pose(position=np.array(pos, dtype=float),
                      orientation=euler_angles_to_quat(np.array([0.0, 0.0, np.radians(yaw_deg)])))
    if scale is not None:
        # [scale]*3 == [scale, scale, scale] -> uniform scale on x, y, z.
        xf.set_local_scale(np.array([scale] * 3, dtype=float))
    return xf


# ===========================================================================
# THE ROOM: floor + four walls, built from plain colored boxes.
# VisualCuboid = a box with geometry + color but NO collider (purely visual).
# `scale` here is the full size in metres. The floor is a thin 20x14x0.1 slab
# sunk 0.05 m so its top sits at z=0 (the ground plane everything rests on).
# ===========================================================================
VisualCuboid(prim_path="/World/Floor", position=np.array([0, 0, -0.05]),
             scale=np.array([2 * HX, 2 * HY, 0.1]), color=np.array([0.42, 0.42, 0.46]))
# Each wall is a thin tall slab centered on one edge of the room, half its
# height up (so it stands on the floor). (name, center[x,y,z], size[x,y,z]):
for name, c, s in [
    ("WallN", [0,  HY, WALL_H / 2], [2 * HX, 0.2, WALL_H]),   # north (+y), spans x
    ("WallS", [0, -HY, WALL_H / 2], [2 * HX, 0.2, WALL_H]),   # south (-y), spans x
    ("WallE", [ HX, 0, WALL_H / 2], [0.2, 2 * HY, WALL_H]),   # east  (+x), spans y
    ("WallW", [-HX, 0, WALL_H / 2], [0.2, 2 * HY, WALL_H]),   # west  (-x), spans y
]:
    VisualCuboid(prim_path=f"/World/{name}", position=np.array(c, dtype=float),
                 scale=np.array(s, dtype=float), color=np.array([0.78, 0.76, 0.71]))

# ---- lighting --------------------------------------------------------------
# Headless interiors render nearly black without explicit lights. Two lights:
#  * DomeLight = soft ambient light from all directions (fills shadows).
#  * DistantLight = a directional "sun" (parallel rays), rotated to come in at
#    an angle so objects cast believable shadows. Intensities are tuned by eye.
dome = UsdLux.DomeLight.Define(stage, "/World/Dome")
dome.CreateIntensityAttr(1400)
sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
sun.CreateIntensityAttr(1600)
# XformCommonAPI.SetRotate sets Euler rotation (degrees) on the sun prim so its
# rays point down and to the side.
UsdGeom.XformCommonAPI(sun.GetPrim()).SetRotate(Gf.Vec3f(-55, 0, 30))

# ===========================================================================
# SHELVING RACKS. Built procedurally from cuboids + a couple of prop refs, so
# we can stamp out many identical racks with one function call each.
# ===========================================================================
ORANGE = np.array([0.85, 0.45, 0.1])
_rack_n = 0   # module-level counter so each rack gets a unique prim path


def build_rack(cx, cy, yaw_deg):
    """Build one 2.4 m-wide, 2-tier pallet rack centered at (cx, cy).

    A rack = 4 vertical posts + 2 horizontal shelf slabs (all cheap
    VisualCuboids) + 2 stocked props (a box and a bin). It's placed in a LOCAL
    frame (rack-relative coordinates) that we rotate by yaw_deg into world
    space via the nested tw() helper — this is how you lay out a sub-assembly
    once and drop rotated copies anywhere.

    Args:
        cx, cy:   world position of the rack's center (metres).
        yaw_deg:  rotation of the whole rack about z (degrees). 0 = the rack's
                  local +x axis points along world +x.
    """
    global _rack_n
    _rack_n += 1
    root = f"/World/Racks/R{_rack_n}"   # unique parent path for this rack
    yr = np.radians(yaw_deg)

    def tw(lx, ly, z):
        """Transform a rack-LOCAL point (lx, ly, z) into WORLD coordinates.

        This is a standard 2D rotation-then-translation. A point (lx, ly)
        rotated by angle yr about the origin becomes:
            x' = lx*cos(yr) - ly*sin(yr)
            y' = lx*sin(yr) + ly*cos(yr)
        Then we add the rack center (cx, cy). z passes through unchanged
        (we never tilt racks). Memorize this pair of formulas — it's THE way
        to place things relative to a rotated parent.
        """
        return np.array([cx + lx * np.cos(yr) - ly * np.sin(yr),
                         cy + lx * np.sin(yr) + ly * np.cos(yr), z])

    # Four uprights at the rack's corners (local x=+/-1.2, y=+/-0.5), each a
    # thin 2.2 m-tall post centered at z=1.1 (so it spans 0..2.2 m).
    for i, (lx, ly) in enumerate([(-1.2, -0.5), (-1.2, 0.5), (1.2, -0.5), (1.2, 0.5)]):
        VisualCuboid(prim_path=f"{root}/Post{i}", position=tw(lx, ly, 1.1),
                     scale=np.array([0.09, 0.09, 2.2]), color=ORANGE)
    # Two shelf slabs at heights 0.75 and 1.6 m. We pass the same yaw as an
    # orientation so the slab is rotated to match the rack (a box's `scale` is
    # axis-aligned, so without the rotation a rotated rack's shelves wouldn't
    # line up with its posts).
    for j, z in enumerate([0.75, 1.6]):
        VisualCuboid(prim_path=f"{root}/Shelf{j}", position=tw(0, 0, z),
                     orientation=euler_angles_to_quat(np.array([0, 0, yr])),
                     scale=np.array([2.5, 1.1, 0.06]), color=np.array([0.33, 0.33, 0.36]))
    # Stock the lower shelf with 2 real props. IMPORTANT BUDGET NOTE: the rack
    # frame is free-ish (VisualCuboids have no collider), but each referenced
    # prop carries collision geometry that the physics engine must parse at
    # startup. An earlier version with ~3x as many props CRASHED PhysX during
    # scene init, so we deliberately keep only 2 props per rack. (Even a
    # kinematic scene that never simulates still pays this parse cost.)
    place(f"{WH}/SM_CardBoxA_01.usd", f"{root}/S0", tw(-0.5, 0, 0.80), yaw_deg + 10)
    place("/Isaac/Props/KLT_Bin/small_KLT.usd", f"{root}/S1", tw(0.5, 0, 0.80), yaw_deg)


# Stamp 6 racks: three along the north wall, three along the south wall, each
# pushed 1.3 m in from the wall so nothing clips. This is the "1 actor -> N
# actors is just data" idea applied to scenery.
for cx in (-6.0, 0.0, 6.0):
    build_rack(cx, HY - 1.3, 0)        # north wall (+y)
    build_rack(cx, -HY + 1.3, 0)       # south wall (-y)

# ---- scattered dressing ----------------------------------------------------
# A few loose props to make the floor feel used. (An earlier version also had
# 9 m pillars and bottle clusters, but those extra collision meshes pushed
# PhysX past its limit at init, so they were cut — see the budget note above.)
place(f"{WH}/SM_BarelPlastic_A_01.usd", "/World/Barrel1", [-8.6, -4.5, 0.0])
place(f"{WH}/SM_BarelPlastic_A_02.usd", "/World/Barrel2", [-8.1, -4.7, 0.0])
place(f"{WH}/SM_BarelPlastic_A_01.usd", "/World/Barrel3", [8.4, 4.6, 0.0], 40)
place("/Isaac/Props/Pallet/pallet.usd", "/World/Pallet1", [7.5, -5.0, 0.0], 20)
place(f"{WH}/SM_CardBoxA_01.usd", "/World/PBox1", [7.5, -5.0, 0.15], 20)  # box on that pallet
place(f"{WH}/SM_FloorDecal_Keepclear.usd", "/World/Decal1", [0.0, 0.0, 0.02])
place(f"{WH}/SM_FloorDecal_Keepclear.usd", "/World/Decal2", [0.0, -3.5, 0.02], 90)

# ===========================================================================
# THE ACTORS: the hero box, the forklift, and the car.
# We keep XFormPrim handles (hero, forklift, car) so we can move them each frame.
# ===========================================================================
BOX_START = np.array([-8.4, 1.5, 0.0])   # where the box sits before pickup
hero = place(f"{WH}/SM_CardBoxA_01.usd", "/World/HeroBox", BOX_START)
place("/Isaac/Props/Forklift/forklift.usd", "/World/Forklift", [0.0, -3.0, 0.0], 0)
forklift = XFormPrim("/World/Forklift")   # handle to move the whole forklift

# The car is the NVIDIA Leatherback (an RC-style vehicle asset). Its .usd is
# authored in CENTIMETRES, so it loads ~40 m long; scale=0.05 brings it to a
# realistic ~2.1 m car. (Confirmed by the report_bbox call below.)
car = place("/Isaac/Robots/NVIDIA/Leatherback/leatherback.usd", "/World/Car",
            [6.0, 0.0, 0.05], scale=0.05)

# Sanity-check the four assets that matter. If any prints EMPTY or a crazy
# size, stop and fix it before wasting a render.
report_bbox("/World/Forklift", "forklift")
report_bbox("/World/Car", "car")
report_bbox("/World/HeroBox", "hero_box")
report_bbox("/World/Racks/R1", "rack")

# ---- find the fork carriage so we can raise/lower the forks ----------------
# The forklift asset is a tree of prims (body, mast, wheels, forks...). To lift
# the box we need the "fork carriage" sub-prim specifically. We walk the
# forklift's subtree (Usd.PrimRange) and pick the first Xformable prim whose
# name ends in "fork". We skip the root itself (its name also contains "fork").
# Lesson: never hardcode sub-prim paths — asset internals differ; search by name.
fork_prim = None
for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/Forklift")):
    if prim.GetPath().pathString == "/World/Forklift":
        continue
    if prim.GetName().lower().endswith("fork") and prim.IsA(UsdGeom.Xformable):
        fork_prim = XFormPrim(prim.GetPath().pathString)
        print("FORK PRIM:", prim.GetPath().pathString)
        break

# ===========================================================================
# CHOREOGRAPHY — the waypoint tables that describe the motion.
# ===========================================================================
# FORKLIFT table rows are (time_s, x, y, yaw_deg, fork_height_m). Between rows
# we interpolate. KEY FACT ABOUT THIS ASSET: its forks point along its LOCAL
# -y axis. So to aim the forks at something in world -x you rotate the forklift
# to yaw = -90 deg; to aim them at world +x you use yaw = +90. (This was found
# by rendering a test frame — never assume which way an asset "faces".)
FORK_WP = [
    (0.0,  0.0, -3.0,    0, 0.05),   # rest, mid-floor, forks low
    (2.0,  0.0, -3.0,    0, 0.05),   # a beat of stillness
    (5.0, -5.5,  1.5,  -90, 0.05),   # drive to the west aisle, turn to face -x
    (7.5, -6.9,  1.5,  -90, 0.05),   # creep forward so the forks go under the box
    (10.0, -6.9, 1.5,  -90, 0.90),   # raise the forks -> lift the box
    (12.5, -5.0, 1.5,  -90, 0.90),   # reverse out, box held high
    (16.0,  0.0, -1.0,  90, 0.90),   # cross the room, rotating to face +x
    (19.0,  5.5, -1.5,  90, 0.90),   # drive to the east aisle
    (21.5,  6.9, -1.5,  90, 0.90),   # creep in over the drop pallet
    (23.5,  6.9, -1.5,  90, 0.15),   # lower the forks -> set the box down
    (26.0,  5.0, -1.5,  90, 0.05),   # back away, forks empty
    (30.0,  0.0, -4.0, 180, 0.05),   # park
]
# The box "rides" the forks only during this time window (roughly, while the
# forks are under it through when it's set down). Outside it, the box keeps
# wherever it was last put.
CARRY_START, CARRY_END = 8.0, 23.5
# How far ahead of the forklift's origin the box sits, along the fork axis.
FORK_FORWARD = 1.5

# CAR table rows are (time_s, x, y) — POSITION ONLY, no yaw. We derive the
# car's heading from its direction of travel instead (see car_pose). This is
# "auto-heading": you describe the *path*, and facing takes care of itself.
CAR_WP = [
    (0.0,  6.0,  0.0), (4.0,  8.0,  4.5), (8.0,  3.0,  5.5),
    (12.0, -3.0, 5.5), (15.0, -6.0, 3.0), (19.0, -6.0, -3.0),
    (23.0, 3.0, -3.5), (27.0, 8.0, -3.0), (30.0, 8.0,  1.0),
]
# If the car ends up driving sideways/backwards, its model's "forward" axis
# differs from local +x; add a fixed offset (e.g. 90 or 180) to fix it here.
CAR_YAW_OFFSET = 0.0


def smoothstep(u):
    """Ease-in/ease-out remap of a 0..1 progress value onto a 0..1 curve.

    Given u going linearly 0 -> 1, returns 3u^2 - 2u^3, which starts and ends
    with zero slope. Interpolating with this instead of raw `u` makes motion
    accelerate smoothly out of a keyframe and decelerate into the next one,
    so vehicles don't teleport-start or jerk to a stop. Its derivative is
    6u(1-u), which is 0 at u=0 and u=1 — that zero slope is the "ease".
    """
    return u * u * (3 - 2 * u)


def interp4(wp, t):
    """Look up a waypoint table at time t, returning the blended pose tuple.

    Works for tables of any width: FORK_WP rows are (t, x, y, yaw, h) so it
    returns (x, y, yaw, h); CAR_WP rows are (t, x, y) so it returns (x, y).

    Algorithm:
      * Before the first keyframe: hold the first pose (no extrapolation).
      * Otherwise find the pair of keyframes (a, b) that straddle t, compute a
        normalized progress u in [0,1] between their times, ease it with
        smoothstep, then linearly blend every pose component: a + u*(b - a).
      * After the last keyframe: hold the last pose.

    Args:
        wp: the waypoint table (list of tuples, first element = time).
        t:  the query time in seconds.
    Returns:
        A tuple of the interpolated pose components (everything after time).
    """
    if t <= wp[0][0]:
        return wp[0][1:]                     # before start: hold first pose
    for a, b in zip(wp, wp[1:]):             # scan adjacent keyframe pairs
        if a[0] <= t <= b[0]:
            u = smoothstep((t - a[0]) / (b[0] - a[0]))   # eased 0..1 progress
            # blend each component i (skip index 0, which is time):
            return tuple(a[i] + u * (b[i] - a[i]) for i in range(1, len(a)))
    return wp[-1][1:]                         # after end: hold last pose


def car_pose(t):
    """Return the car's (x, y, yaw) at time t, with yaw AUTO-derived.

    We read the car's position now, and its position a hair in the future
    (t + 0.25 s), then point the car along the vector between them:
        yaw = atan2(dy, dx)
    atan2 is the "angle of a 2D vector" — it returns the heading (radians) of
    the direction the car is about to move. So the car always looks where it's
    going, no matter how the path curves, with zero manual yaw tuning. The
    `if (dx or dy)` guards the degenerate case where the car isn't moving
    (both deltas zero) so atan2(0,0) doesn't give a meaningless angle. Finally
    we add CAR_YAW_OFFSET to correct for the model's forward-axis convention.
    """
    x, y = interp4(CAR_WP, t)
    xa, ya = interp4(CAR_WP, min(t + 0.25, CAR_WP[-1][0]))  # a look-ahead point
    yaw = np.arctan2(ya - y, xa - x) if (xa - x or ya - y) else 0.0
    return x, y, yaw + np.radians(CAR_YAW_OFFSET)


def apply_frame(t):
    """Pose EVERY animated object for the single instant t. Called once/frame.

    This is the heart of the animation. It:
      1. Places and orients the forklift from FORK_WP.
      2. Raises/lowers the fork carriage (a sub-prim) to the keyframed height.
      3. During the carry window, "attaches" the box by setting its pose to the
         forklift's pose plus an offset along the fork axis (the pose-follow
         trick — cheaper and more robust than real physics grabbing/parenting).
      4. Places and orients the car (with auto-heading).

    Args:
        t: current time in seconds.
    """
    # --- forklift body ---
    fx, fy, fyaw, fh = interp4(FORK_WP, t)
    fyr = np.radians(fyaw)
    forklift.set_world_pose(position=np.array([fx, fy, 0.0]),
                            orientation=euler_angles_to_quat(np.array([0, 0, fyr])))
    # --- fork carriage height (a LOCAL move within the forklift) ---
    if fork_prim is not None:
        p, q = fork_prim.get_local_pose()   # current local translation p, rotation q
        # keep x,y of the forks, override the z (height) with the keyframed fh:
        fork_prim.set_local_pose(translation=np.array([p[0], p[1], fh]), orientation=q)
    # --- the box "rides" the forks during the carry window ---
    if CARRY_START <= t <= CARRY_END:
        # The forks point along the forklift's local -y. In world space, a
        # local -y direction after a yaw rotation is (sin(yaw), -cos(yaw)). So
        # the box sits FORK_FORWARD metres along that direction from the
        # forklift origin, at the current fork height fh:
        bx = fx + FORK_FORWARD * np.sin(fyr)
        by = fy - FORK_FORWARD * np.cos(fyr)
        hero.set_world_pose(position=np.array([bx, by, fh]),
                            orientation=euler_angles_to_quat(np.array([0, 0, fyr])))
    # --- the car ---
    cx, cy, cyaw = car_pose(t)
    car.set_world_pose(position=np.array([cx, cy, 0.05]),
                       orientation=euler_angles_to_quat(np.array([0, 0, cyaw])))


# ===========================================================================
# CAMERA. One camera prim we reposition every frame to orbit the room.
# ===========================================================================
# Create the camera at 12 m up (initial spot; place_camera moves it each frame).
camera = Camera(prim_path="/World/Cam", position=np.array([0, 0, 12]),
                frequency=FPS, resolution=(1280, 720))
camera.initialize()
# world.reset() commits the scene to the renderer/physics and must run before
# we render. We call camera.initialize() again after it because reset can
# invalidate the earlier handle.
world.reset()
camera.initialize()
# Isaac's default camera is telephoto (~24 deg field of view) — too tight for a
# whole-room shot. We widen it by shortening the focal length (smaller focal
# length = wider view). We write the USD attribute directly because the Python
# setter crashed in this build. f-stop 0 disables depth-of-field blur so the
# whole scene is sharp.
usd_cam = UsdGeom.Camera(stage.GetPrimAtPath("/World/Cam"))
usd_cam.GetFocalLengthAttr().Set(usd_cam.GetFocalLengthAttr().Get() * 0.42)  # ~2.4x wider
usd_cam.GetFStopAttr().Set(0.0)

CENTER = np.array([0.0, 0.0, 0.8])   # the point the camera always looks at


def place_camera(t):
    """Move the camera to a slow elevated orbit around CENTER at time t.

    * theta sweeps the orbit angle: it starts at 215 deg and advances with
      time. The `* 0.5` makes the camera travel only half a full circle over
      the whole clip (a gentle drift, not a dizzying spin).
    * pos puts the camera on an ellipse (13.5 m in x, 10 m in y) at height 11 m
      — above the 5 m walls, so it always sees down into the room.
    * We then aim it at CENTER by computing a look-at rotation:
        d     = direction from camera to target
        yaw   = atan2(d.y, d.x)                       (heading in the xy-plane)
        pitch = atan2(-d.z, horizontal_distance)      (tilt down toward target)
      and convert (roll=0, pitch, yaw) to a quaternion. This is the standard
      "point a camera at a target" recipe.
    """
    theta = np.radians(215) + 2 * np.pi * (t / (NUM_FRAMES / FPS)) * 0.5
    pos = CENTER + np.array([13.5 * np.cos(theta), 10.0 * np.sin(theta), 11.0])
    d = CENTER - pos                              # look direction
    yaw = np.arctan2(d[1], d[0])                  # turn toward the target
    pitch = np.arctan2(-d[2], np.linalg.norm(d[:2]))  # tilt down toward it
    camera.set_world_pose(position=pos,
                          orientation=euler_angles_to_quat(np.array([0.0, pitch, yaw])))


# ---- prepare the output folder ---------------------------------------------
os.makedirs(FRAMES_DIR, exist_ok=True)
for f in os.listdir(FRAMES_DIR):        # clear any frames from a previous run
    os.remove(os.path.join(FRAMES_DIR, f))

# ---- warm up the renderer --------------------------------------------------
# The RTX renderer streams textures/compiles shaders lazily; the first few
# frames are incomplete. We pose t=0 and render ~40 throwaway frames so the
# real capture starts on a fully-loaded image.
print("Warming up renderer...")
apply_frame(0.0)
place_camera(0.0)
for _ in range(40):
    world.render()


def capture(path):
    """Grab the current camera image and write it to `path` as a JPEG.

    camera.get_rgba() returns an (H, W, 4) uint8 array (red, green, blue,
    alpha). We drop alpha ([:, :, :3]), and convert RGB->BGR because OpenCV's
    imwrite expects BGR channel order. Returns True on success, False if the
    renderer handed back an empty frame (which happens if the render pipeline
    isn't ready — a useful signal that something is wrong).
    """
    img = camera.get_rgba()
    if img is not None and img.size > 0:
        cv2.imwrite(path, cv2.cvtColor(img[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR))
        return True
    return False


# ===========================================================================
# MAIN LOOP. Two modes:
#   SMOKE=1  -> render a few stills at key story beats (fast sanity check).
#   default  -> render every frame of the full video.
# In BOTH: for each time t we pose the world (apply_frame), move the camera
# (place_camera), render, and save the pixels. Note there is NO world.step():
# this is animation (we drive poses), not physics simulation.
# ===========================================================================
if SMOKE:
    # Sample the moments that matter: start, forks-under-box, lift, mid-cross,
    # arrive-at-drop, and a late patrol frame. Print each actor's pose so you
    # can verify the choreography numerically, not just visually.
    for t in [0.0, 7.5, 10.0, 16.0, 21.5, 27.0]:
        apply_frame(t)
        place_camera(t)
        for _ in range(15):          # a few renders to settle the image
            world.render()
        fx, fy, fyaw, fh = interp4(FORK_WP, t)
        cx, cy, _ = car_pose(t)
        ok = capture(f"/workspace/smoke_xl_t{int(t * 10):03d}.jpg")
        print(f"SMOKE t={t}: {'saved' if ok else 'EMPTY'} "
              f"fork=({fx:.1f},{fy:.1f},yaw{fyaw:.0f},h{fh:.2f}) car=({cx:.1f},{cy:.1f})")
else:
    saved = 0
    for i in range(NUM_FRAMES):
        t = i / FPS                  # convert frame index -> seconds
        apply_frame(t)
        place_camera(t)
        world.render()
        if capture(os.path.join(FRAMES_DIR, f"frame_{i:04d}.jpg")):  # zero-padded name for ffmpeg
            saved += 1
        if (i + 1) % 100 == 0:
            print(f"Recorded {i + 1}/{NUM_FRAMES} frames")
    print(f"Done. Saved {saved}/{NUM_FRAMES} frames to {FRAMES_DIR}")

# Cleanly shut Kit down (flushes the renderer, releases the GPU). Always do
# this or the process can hang holding the GPU.
simulation_app.close()
