"""Warehouse forklift demo v2 — PHYSICS-DRIVEN. Isaac Sim 5.1, headless.

What changed vs v1 (warehouse_forklift.py):
  v1 was pure kinematics: every pose scripted, world.render() only, the box
  "attached" to the forks by pose-copying. v2 turns the physics engine on:

  * The hero box is a DYNAMIC RIGID BODY: gravity pulls it, colliders stop
    it, friction holds it on the forks. Nobody sets its pose after spawn.
  * The forklift is a KINEMATIC rigid body: we still drive it from the
    waypoint table (animation), but its colliders push dynamic objects.
    Kinematic = "infinitely strong animated object" — it moves the world,
    the world can't move it.
  * Racks, barrels, pallets, walls are STATIC colliders: immovable scenery.
  * The loop calls world.step(render=True): physics advances, then renders.

  The pickup now works like reality: forks slide under the box, lift it
  (contact + friction carry it), and at aisle B the box is lowered to the
  floor; when the forklift reverses, floor friction holds the box while the
  forks slide out. No attach/detach code at all — physics does it.

Env vars:
  SMOKE=1     render stills at key waypoints instead of the video
  NUM_FRAMES  override frame count (default 520 = 26 s at 20 fps)
"""
import os
import numpy as np
import cv2

print("Starting Isaac Sim (headless)...")
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

from omni.isaac.core import World
from omni.isaac.core.objects import FixedCuboid
from omni.isaac.core.prims import XFormPrim, RigidPrim
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.sensor import Camera
from omni.physx.scripts import utils as physx_utils
from omni.physx.scripts import physicsUtils
import omni
from pxr import Usd, UsdGeom, UsdLux, UsdShade, UsdPhysics, PhysxSchema, Gf, Sdf

FRAMES_DIR = "/workspace/warehouse2_frames"
FPS = 20
NUM_FRAMES = int(os.environ.get("NUM_FRAMES", str(29 * FPS)))
SMOKE = os.environ.get("SMOKE") == "1"
WARMUP_STEPS = 40

# Physics steps at 60 Hz (stable), we render every 3rd step (20 fps).
world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0, rendering_dt=1.0 / 20.0)
stage = omni.usd.get_context().get_stage()
assets_root = get_assets_root_path()
bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])

WH = "/Isaac/Environments/Simple_Warehouse/Props"


def report_bbox(prim_path, label):
    rng = bbox_cache.ComputeWorldBound(stage.GetPrimAtPath(prim_path)).ComputeAlignedRange()
    if rng.IsEmpty():
        print(f"BBOX {label}: EMPTY (asset failed to load!)")
        return None
    size = rng.GetMax() - rng.GetMin()
    print(f"BBOX {label}: size=({size[0]:.2f}, {size[1]:.2f}, {size[2]:.2f}) "
          f"min_z={rng.GetMin()[2]:.2f}")
    return rng


def place(usd_path, prim_path, pos, yaw_deg=0.0, scale=None):
    add_reference_to_stage(usd_path=assets_root + usd_path, prim_path=prim_path)
    xf = XFormPrim(prim_path)
    xf.set_world_pose(position=np.array(pos, dtype=float),
                      orientation=euler_angles_to_quat(np.array([0.0, 0.0, np.radians(yaw_deg)])))
    if scale is not None:
        xf.set_local_scale(np.array([scale] * 3, dtype=float))
    return xf


def make_static(prim_path):
    """Static collider: immovable scenery the physics engine collides against.
    'none' approximation = use the actual mesh (accurate, fine for statics)."""
    physx_utils.setStaticCollider(stage.GetPrimAtPath(prim_path), approximationShape="none")


def make_dynamic(prim_path, approx="convexHull"):
    """Dynamic rigid body: gravity + collisions move it. Convex hull for
    simple shapes; convexDecomposition when concavities matter (a pallet's
    fork slots must stay open, or the forks can't enter them)."""
    physx_utils.setRigidBody(stage.GetPrimAtPath(prim_path), approx, False)


def make_kinematic(prim_path):
    """Kinematic rigid body: animated by us, collides with dynamics.

    CRITICAL: convexDecomposition, NOT convexHull. A convex hull fills every
    concavity — the gap between the fork tines becomes solid, and the
    'invisible wedge' bulldozes the box instead of sliding under it. Convex
    decomposition approximates the true concave shape with multiple hulls,
    so the forks stay separate prongs. (v2 smoke test found this the hard
    way: the box was launched across the room.)"""
    physx_utils.setRigidBody(stage.GetPrimAtPath(prim_path), "convexDecomposition", True)


# ---------------------------------------------------------------- the room --
ROOM = 10.0
# FixedCuboid (not VisualCuboid) — it has a collider, so the floor actually
# holds things up now that gravity exists.
FixedCuboid(prim_path="/World/Floor", position=np.array([0, 0, -0.05]),
            scale=np.array([ROOM, ROOM, 0.1]), color=np.array([0.45, 0.45, 0.48]))
for name, c, s in [
    ("WallN", [0,  ROOM / 2, 5], [ROOM, 0.15, ROOM]),
    ("WallS", [0, -ROOM / 2, 5], [ROOM, 0.15, ROOM]),
    ("WallE", [ ROOM / 2, 0, 5], [0.15, ROOM, ROOM]),
    ("WallW", [-ROOM / 2, 0, 5], [0.15, ROOM, ROOM]),
]:
    FixedCuboid(prim_path=f"/World/{name}", position=np.array(c, dtype=float),
                scale=np.array(s, dtype=float), color=np.array([0.75, 0.73, 0.68]))

dome = UsdLux.DomeLight.Define(stage, "/World/Dome")
dome.CreateIntensityAttr(1500)
sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
sun.CreateIntensityAttr(1500)
UsdGeom.XformCommonAPI(sun.GetPrim()).SetRotate(Gf.Vec3f(-55, 0, 25))

# ------------------------------------------------- racks, cargo, obstacles --
# Loaded rack piles along the E and W walls (the "aisles"), a rack frame in a
# corner, barrel clusters and a stray pallet as obstacles the forklift path
# must respect. Everything here is a static collider.
statics = []

# The bucket has no pre-assembled loaded rack (SM_RackPile_* turned out to be
# piles of disassembled beams — the 0.3 m bbox gave it away), so we build
# two-tier shelving from cuboids and stock it with boxes and bins.
ORANGE = np.array([0.85, 0.45, 0.1])
rack_n = 0


def build_rack(cx, cy, yaw_deg, stock=True):
    """A 2.4 m wide, 2-tier pallet rack: 4 uprights + 2 shelf slabs, stocked
    with cardboard boxes / KLT bins. All FixedCuboid = static colliders."""
    global rack_n
    rack_n += 1
    root = f"/World/Racks/R{rack_n}"
    yaw_r = np.radians(yaw_deg)
    # local rack frame: width along local x, depth along local y
    def to_world(lx, ly, z):
        return np.array([cx + lx * np.cos(yaw_r) - ly * np.sin(yaw_r),
                         cy + lx * np.sin(yaw_r) + ly * np.cos(yaw_r), z])
    for i, (lx, ly) in enumerate([(-1.2, -0.5), (-1.2, 0.5), (1.2, -0.5), (1.2, 0.5)]):
        FixedCuboid(prim_path=f"{root}/Post{i}", position=to_world(lx, ly, 1.1),
                    scale=np.array([0.09, 0.09, 2.2]), color=ORANGE)
    for j, z in enumerate([0.75, 1.6]):
        # FixedCuboid scale is axis-aligned — pass the yaw as an orientation
        # or the shelf slab juts into the aisle (v11 bug: unrotated shelf
        # hovered over the pickup point and blocked the lift)
        FixedCuboid(prim_path=f"{root}/Shelf{j}", position=to_world(0, 0, z),
                    orientation=euler_angles_to_quat(np.array([0.0, 0.0, yaw_r])),
                    scale=np.array([2.5, 1.1, 0.06]), color=np.array([0.35, 0.35, 0.38]))
    if stock:
        place(f"{WH}/SM_CardBoxA_01.usd", f"{root}/Stock0", to_world(-0.6, 0, 0.80), yaw_deg=yaw_deg + 10)
        place("/Isaac/Props/KLT_Bin/small_KLT.usd", f"{root}/Stock1", to_world(0.5, -0.2, 0.80), yaw_deg=yaw_deg)
        place("/Isaac/Props/KLT_Bin/small_KLT.usd", f"{root}/Stock2", to_world(0.7, 0.2, 0.80), yaw_deg=yaw_deg)
        place(f"{WH}/SM_CardBoxA_01.usd", f"{root}/Stock3", to_world(0.2, 0, 1.66), yaw_deg=yaw_deg - 15)
        place("/Isaac/Props/Pallet/pallet.usd", f"{root}/Stock4", to_world(-0.7, 0, 1.63), yaw_deg=yaw_deg)
    return root


# aisle A (west wall): two stocked racks. aisle B (east wall): one stocked +
# one empty rack (the drop zone). Rack colliders come from their FixedCuboids;
# only the stocked props need static colliders.
rack_stock_paths = []
for cx, cy, yaw, stock in [(-4.2, -1.6, 90, True), (-4.2, 1.6, 90, True),
                           (4.2, -1.6, -90, True), (4.2, 1.6, -90, False)]:
    root = build_rack(cx, cy, yaw, stock)
    if stock:
        for k in range(5):
            rack_stock_paths.append(f"{root}/Stock{k}")

# back wall: a pillar and the beam pile reused honestly — as floor clutter
statics.append(place(f"{WH}/SM_PillarA_9M.usd", "/World/Pillar", [1.8, 4.6, 0.0]))
statics.append(place(f"{WH}/SM_RackPile_04.usd", "/World/Obst/BeamPile", [-1.6, 4.2, 0.0], yaw_deg=30))

# obstacles on the floor
statics.append(place(f"{WH}/SM_BarelPlastic_A_01.usd", "/World/Obst/Barrel1", [-1.2, 2.6, 0.0]))
statics.append(place(f"{WH}/SM_BarelPlastic_A_02.usd", "/World/Obst/Barrel2", [-0.7, 2.9, 0.0]))
statics.append(place("/Isaac/Props/Pallet/pallet.usd", "/World/Obst/Pallet1", [1.6, -2.6, 0.0], yaw_deg=25))
statics.append(place(f"{WH}/SM_CardBoxA_01.usd", "/World/Obst/PalletBox", [1.6, -2.6, 0.15], yaw_deg=25))

# floor decals (pure decoration — no colliders needed)
place(f"{WH}/SM_FloorDecal_Keepclear.usd", "/World/Decals/D1", [0.0, 0.0, 0.01])
place(f"{WH}/SM_FloorDecal_StripeFull_4m.usd", "/World/Decals/D2", [-2.6, 0.0, 0.01], yaw_deg=90)
place(f"{WH}/SM_FloorDecal_StripeFull_4m.usd", "/World/Decals/D3", [2.6, 0.0, 0.01], yaw_deg=90)

# ------------------------------------------------------- dynamic objects ----
# The hero load. v3 lesson: fork tines can't slide under a solid box resting
# flat on the floor — they hit its face and bulldoze it.
# DUNNAGE STAGING (v12 scope decision): a dynamic pallet pickup needs
# contact-offset tuning we're deferring; the demo goal is moving the BOX.
# Warehouses stage un-palletized loads on dunnage blocks — two low static
# beams under the box ends, sitting in the center strip the blades never
# enter. The box bottom sits at 0.16; blades enter at z=[0.057,0.107] with
# a designed 5 cm gap (beyond PhysX contact-offset phantom range), then
# rise to lift the box off the blocks.
for i, bx in enumerate([-3.28, -2.72]):
    FixedCuboid(prim_path=f"/World/Dunnage{i}", position=np.array([bx, -0.2, 0.08]),
                scale=np.array([0.12, 0.22, 0.16]), color=np.array([0.45, 0.32, 0.18]))
BOX_START = np.array([-3.0, -0.2, 0.24])
place(f"{WH}/SM_CardBoxA_01.usd", "/World/HeroBox", BOX_START)

# a loose stack next to aisle A that can topple if clipped — physics showcase
place(f"{WH}/SM_CardBoxA_01.usd", "/World/Loose/Stack1", [-2.6, 3.2, 0.05])
place(f"{WH}/SM_CardBoxA_01.usd", "/World/Loose/Stack2", [-2.6, 3.2, 0.60], yaw_deg=20)

# ---------------------------------------------------------------- forklift --
place("/Isaac/Props/Forklift/forklift.usd", "/World/Forklift", [0.0, -3.6, 0.0], yaw_deg=90)
forklift = XFormPrim("/World/Forklift")

report_bbox("/World/Forklift", "forklift")
report_bbox("/World/HeroBox", "hero_box")
report_bbox("/World/Obst/Barrel1", "barrel")

fork_prim = None
for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/Forklift")):
    if prim.GetPath().pathString == "/World/Forklift":
        continue
    if prim.GetName().lower().endswith("fork") and prim.IsA(UsdGeom.Xformable):
        fork_prim = XFormPrim(prim.GetPath().pathString)
        print("FORK PRIM:", prim.GetPath().pathString)
        break

# --------------------------------------------------------- physics set-up --
for xf in statics:
    make_static(xf.prim_path)
for p in rack_stock_paths:
    make_static(p)
make_static("/World/Obst/PalletBox")          # box glued to obstacle pallet
make_dynamic("/World/HeroBox")
make_dynamic("/World/Loose/Stack1")
make_dynamic("/World/Loose/Stack2")
make_kinematic("/World/Forklift")

# v10 lesson: the fork's two blades are ONE mesh, and convex decomposition
# bridges the gap between them into a solid wall at blade height — the
# actual plow in every failed pickup. Cure like the pallet: kill collision
# on the fork mesh, hand-author two blade boxes in the measured lanes.
# v11 lesson: with honest blades the pallet STILL plowed — the body/mast
# decomposition was a second plow. Only hand-authored geometry can be
# trusted near the pallet: disable collision on EVERY forklift mesh; the
# two blade boxes below become the forklift's only colliders.
for m in Usd.PrimRange(stage.GetPrimAtPath("/World/Forklift")):
    if m.IsA(UsdGeom.Mesh):
        UsdPhysics.CollisionAPI.Apply(m).CreateCollisionEnabledAttr(False)

if fork_prim is not None:

    def blade(name, lx):
        c = UsdGeom.Cube.Define(stage, f"{fork_prim.prim_path}/Blade_{name}")
        c.GetSizeAttr().Set(1.0)
        api = UsdGeom.XformCommonAPI(c.GetPrim())
        api.SetTranslate(Gf.Vec3d(lx, -1.2, 0.082))   # fork-local; verified via smoke dump
        api.SetScale(Gf.Vec3f(0.12, 1.1, 0.05))
        UsdPhysics.CollisionAPI.Apply(c.GetPrim())
        UsdGeom.Imageable(c.GetPrim()).MakeInvisible()

    blade("L", -0.32)
    blade("R", 0.32)

# Extra insurance on the hero box: continuous collision detection stops a
# fast-moving collider from tunneling straight through thin fork tines.
PhysxSchema.PhysxRigidBodyAPI.Apply(stage.GetPrimAtPath("/World/HeroBox")).CreateEnableCCDAttr(True)

# High-friction physics material so the load doesn't slide off during the
# carry. Friction is THE tuning knob here. Defining the material is not
# enough — it must be BOUND to the collider prims (v2 lesson: an unbound
# material is a silent no-op).
mat = UsdShade.Material.Define(stage, "/World/Physics/GripMat")
grip = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
grip.CreateStaticFrictionAttr(1.2)
grip.CreateDynamicFrictionAttr(1.1)
grip.CreateRestitutionAttr(0.0)
for p in ["/World/HeroBox", "/World/Forklift", "/World/Floor",
          "/World/Dunnage0", "/World/Dunnage1",
          "/World/Forklift/S_ForkliftFork/Blade_L",
          "/World/Forklift/S_ForkliftFork/Blade_R"]:
    physicsUtils.add_physics_material_to_prim(stage, stage.GetPrimAtPath(p),
                                              Sdf.Path("/World/Physics/GripMat"))

# ------------------------------------------------------------ choreography --
# Same waypoint idea as v1, but slower: a kinematic body that accelerates hard
# launches whatever rests on its forks. Gentle motion = stable load.
# Fork axis is local -y => fork_direction_angle = yaw - 90 (verified in v1).
# Fork tips reach ~2.1 m ahead of the forklift origin (measured from smoke
# stills). The square-up spot must leave the tips CLEAR of the box before the
# straight-line creep — v2's square-up at x=-1.2 put the tips through the box
# and the turn swept it away like a broom.
WAYPOINTS = [
    (0.0,  0.0, -3.6,   90, 0.05),   # rest, facing +x
    (2.0,  0.0, -3.6,   90, 0.05),   # beat of stillness (physics settles)
    (5.0,  0.0, -0.2,    0, 0.0),    # drive up into the corridor
    # measured: blade z = [fh+0.057, fh+0.107]; box bottom on dunnage = 0.16.
    # fh=0.0 sends the blades through the 5 cm gap under the box.
    (8.0, -0.4, -0.2,  -90, 0.0),    # square up, blades at gap height
    (12.0, -1.65, -0.2, -90, 0.0),   # EXTRA-slow creep under the box
    (15.0, -1.65, -0.2, -90, 0.55),  # SLOW lift: box rises off the blocks
    (18.0, -0.4, -0.2,  -90, 0.55),  # back out with the load
    (22.0,  0.4,  0.9,   90, 0.55),  # 4-second 180-degree turn (v12: fast turn threw the load)
    (25.0,  1.55, 0.9,   90, 0.55),  # approach aisle B drop zone
    (27.0,  1.55, 0.9,   90, -0.08), # lower until the box grounds
    (29.0,  0.8,  0.9,   90, -0.08), # reverse: floor friction strips the box off
]


def smoothstep(u):
    return u * u * (3 - 2 * u)


def pose_at(t):
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
        p, q = fork_prim.get_local_pose()
        fork_prim.set_local_pose(translation=np.array([p[0], p[1], fh]), orientation=q)
    # NOTE: no box code! The box is a dynamic rigid body — physics moves it.


# ---------------------------------------------------------------- camera ----
camera = Camera(prim_path="/World/Cam", position=np.array([4.2, -4.2, 5.2]),
                frequency=FPS, resolution=(1280, 720))
camera.initialize()
world.reset()
camera.initialize()

usd_cam = UsdGeom.Camera(stage.GetPrimAtPath("/World/Cam"))
usd_cam.GetFocalLengthAttr().Set(usd_cam.GetFocalLengthAttr().Get() * 0.4)
usd_cam.GetFStopAttr().Set(0.0)

CAM_LOOK = np.array([0.0, 0.0, 0.6])
CAM_POS = np.array([4.2, -4.2, 5.2])
d = CAM_LOOK - CAM_POS
camera.set_world_pose(position=CAM_POS,
                      orientation=euler_angles_to_quat(np.array([
                          0.0, np.arctan2(-d[2], np.linalg.norm(d[:2])), np.arctan2(d[1], d[0])])))

os.makedirs(FRAMES_DIR, exist_ok=True)
for f in os.listdir(FRAMES_DIR):
    os.remove(os.path.join(FRAMES_DIR, f))

print("Warming up (physics settles spawned bodies)...")
# RigidPrim reads the PHYSICS-side pose. The USD bbox cache lies here: with
# Fabric enabled, simulated poses never write back to USD (v2 lesson — the
# 'settled' bbox looked frozen at spawn height even when physics worked).
hero_rigid = RigidPrim("/World/HeroBox")
apply_frame(0.0)
for _ in range(WARMUP_STEPS):
    world.step(render=True)
pos, _ = hero_rigid.get_world_pose()
print(f"HERO SETTLED at ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.3f}) "
      f"(spawned z=0.10; ~0.0x means gravity+floor collider work)")


def capture(path):
    img = camera.get_rgba()
    if img is not None and img.size > 0:
        cv2.imwrite(path, cv2.cvtColor(img[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR))
        return True
    return False


if SMOKE:
    # NB: in physics mode we can't jump in time (the box's state is history-
    # dependent), so smoke mode fast-forwards by stepping without capture.
    checkpoints = [0.0, 8.0, 12.0, 15.0, 18.0, 22.0, 27.0, 29.0]
    t, i = 0.0, 0
    for target in checkpoints:
        while t < target:
            apply_frame(t)
            world.step(render=False)
            t += 1.0 / 60.0
        for _ in range(5):
            world.step(render=True)
        ok = capture(f"/workspace/smoke2_t{int(target*10):03d}.jpg")
        hp, _ = hero_rigid.get_world_pose()
        print(f"SMOKE t={target}: {'saved' if ok else 'EMPTY'} "
              f"box=({hp[0]:.2f}, {hp[1]:.2f}, {hp[2]:.2f})")
        # measure the tines: world bbox of the fork part (USD-side pose is
        # valid here because WE move it). Tine bottom z + tip reach = the
        # numbers that must line up with the pallet slots.
        if fork_prim is not None:
            bbox_cache.Clear()
            fr = bbox_cache.ComputeWorldBound(
                stage.GetPrimAtPath(fork_prim.prim_path)).ComputeAlignedRange()
            if not fr.IsEmpty():
                print(f"  FORKBBOX x=[{fr.GetMin()[0]:.2f},{fr.GetMax()[0]:.2f}] "
                      f"z=[{fr.GetMin()[2]:.3f},{fr.GetMax()[2]:.3f}]")
            if target == 8.0:
                # dump every mesh in the fork assembly: reveals the tines'
                # exact lateral (y) positions and thickness — stop guessing
                for mp in Usd.PrimRange(stage.GetPrimAtPath(fork_prim.prim_path)):
                    if mp.IsA(UsdGeom.Mesh) or mp.IsA(UsdGeom.Cube):
                        bbox_cache.Clear()
                        mr = bbox_cache.ComputeWorldBound(mp).ComputeAlignedRange()
                        if not mr.IsEmpty():
                            print(f"  FORKMESH {mp.GetName()}: "
                                  f"x=[{mr.GetMin()[0]:.2f},{mr.GetMax()[0]:.2f}] "
                                  f"y=[{mr.GetMin()[1]:.2f},{mr.GetMax()[1]:.2f}] "
                                  f"z=[{mr.GetMin()[2]:.3f},{mr.GetMax()[2]:.3f}]")
else:
    saved = 0
    step_per_frame = 3   # 60 Hz physics, 20 fps video
    for i in range(NUM_FRAMES):
        t = i / FPS
        for s in range(step_per_frame):
            apply_frame(t + s / 60.0)
            world.step(render=(s == step_per_frame - 1))
        if capture(os.path.join(FRAMES_DIR, f"frame_{i:04d}.jpg")):
            saved += 1
        if (i + 1) % 50 == 0:
            print(f"Recorded {i + 1}/{NUM_FRAMES} frames")
    print(f"Done. Saved {saved}/{NUM_FRAMES} frames to {FRAMES_DIR}")

simulation_app.close()
