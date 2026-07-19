"""Sensor rig demo — Isaac Sim 5.1, headless.

One camera, three synchronized sensor outputs rendered side by side:
RGB | depth (distance_to_camera) | semantic segmentation.

WHAT THIS TEACHES:
* Replicator ANNOTATORS: the renderer can produce more than pretty pixels.
  A "render product" is a camera+resolution binding; annotators attach to it
  and each one extracts a different ground-truth signal from the same frame.
  This is the machinery behind every synthetic dataset.
* SEMANTICS: segmentation only knows what you label. `add_update_semantics`
  stamps a class name on a prim subtree; unlabeled geometry falls into the
  background class. Labels are data you author, not something inferred.
* The scene + choreography are the kinematic warehouse demo (v1) — proven
  parts reused so the new concept (sensors) is the only variable.

Output: 600 frames (30 s @ 20 fps) of 1920x360 3-panel composites in
/workspace/sensor_frames, encoded to sensor_rig.mp4 by the runner command.

Run (on the pod):
  export OMNI_KIT_ACCEPT_EULA=yes
  xvfb-run -a -s "-screen 0 1280x720x24" \
      /workspace/isaac_env/bin/python3 -u sensor_rig.py
  ffmpeg -framerate 20 -i /workspace/sensor_frames/frame_%04d.jpg \
         -c:v libx264 -pix_fmt yuv420p sensor_rig.mp4
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
from omni.isaac.core.utils.semantics import add_update_semantics
from omni.isaac.sensor import Camera
import omni.replicator.core as rep
import omni
from pxr import Usd, UsdGeom, UsdLux, Gf

FRAMES_DIR = "/workspace/sensor_frames"
FPS = 20
NUM_FRAMES = int(os.environ.get("NUM_FRAMES", str(30 * FPS)))
SMOKE = os.environ.get("SMOKE") == "1"
PANEL_W, PANEL_H = 640, 360
DEPTH_MAX = 14.0   # meters; depth is normalized against this for colormap

world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()
assets_root = get_assets_root_path()

WH = "/Isaac/Environments/Simple_Warehouse/Props"


def place(usd_path, prim_path, pos, yaw_deg=0.0, scale=None):
    add_reference_to_stage(usd_path=assets_root + usd_path, prim_path=prim_path)
    xf = XFormPrim(prim_path)
    xf.set_world_pose(position=np.array(pos, dtype=float),
                      orientation=euler_angles_to_quat(np.array([0.0, 0.0, np.radians(yaw_deg)])))
    if scale is not None:
        xf.set_local_scale(np.array([scale] * 3, dtype=float))
    return xf


def label(prim_path, class_name):
    """Semantic label: this is what the segmentation annotator reports.
    No label = background. Labels live on prims like any other USD data."""
    add_update_semantics(stage.GetPrimAtPath(prim_path), class_name)


# ------------------------------------------------ scene (kinematic v1 set) --
ROOM = 10.0
VisualCuboid(prim_path="/World/Floor", position=np.array([0, 0, -0.05]),
             scale=np.array([ROOM, ROOM, 0.1]), color=np.array([0.45, 0.45, 0.48]))
for name, c, s in [
    ("WallN", [0,  ROOM / 2, 5], [ROOM, 0.15, ROOM]),
    ("WallS", [0, -ROOM / 2, 5], [ROOM, 0.15, ROOM]),
    ("WallE", [ ROOM / 2, 0, 5], [0.15, ROOM, ROOM]),
    ("WallW", [-ROOM / 2, 0, 5], [0.15, ROOM, ROOM]),
]:
    VisualCuboid(prim_path=f"/World/{name}", position=np.array(c, dtype=float),
                 scale=np.array(s, dtype=float), color=np.array([0.75, 0.73, 0.68]))
    label(f"/World/{name}", "wall")
label("/World/Floor", "floor")

dome = UsdLux.DomeLight.Define(stage, "/World/Dome")
dome.CreateIntensityAttr(1500)
sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
sun.CreateIntensityAttr(1500)
UsdGeom.XformCommonAPI(sun.GetPrim()).SetRotate(Gf.Vec3f(-55, 0, 25))

aisle_x = {"A": -3.9, "B": 3.9}
pallet_ys = [-3.0, -1.0, 1.0, 3.0]
n = 0
for aisle, ax in aisle_x.items():
    for y in pallet_ys:
        n += 1
        place("/Isaac/Props/Pallet/pallet.usd", f"/World/Aisles/Pallet_{n}", [ax, y, 0.0], yaw_deg=90)
        label(f"/World/Aisles/Pallet_{n}", "pallet")
        if aisle == "B" and y == 1.0:
            continue
        if n % 2 == 0:
            for tag, off in [("a", -0.2), ("b", 0.2)]:
                place("/Isaac/Props/KLT_Bin/small_KLT.usd", f"/World/Aisles/Load_{n}{tag}", [ax, y + off, 0.30])
                label(f"/World/Aisles/Load_{n}{tag}", "bin")
        else:
            place(f"{WH}/SM_CardBoxA_01.usd", f"/World/Aisles/Load_{n}a", [ax, y, 0.30], yaw_deg=15)
            label(f"/World/Aisles/Load_{n}a", "box")

place(f"{WH}/SM_BarelPlastic_A_01.usd", "/World/Barrel1", [-1.2, 2.6, 0.0])
place(f"{WH}/SM_BarelPlastic_A_02.usd", "/World/Barrel2", [-0.7, 2.9, 0.0])
label("/World/Barrel1", "barrel")
label("/World/Barrel2", "barrel")

BOX_START = np.array([-2.9, -1.0, 0.0])
hero = place(f"{WH}/SM_CardBoxA_01.usd", "/World/HeroBox", BOX_START)
label("/World/HeroBox", "box")

place("/Isaac/Props/Forklift/forklift.usd", "/World/Forklift", [0.0, 3.5, 0.0], yaw_deg=0)
forklift = XFormPrim("/World/Forklift")
label("/World/Forklift", "forklift")

fork_prim = None
for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/Forklift")):
    if prim.GetPath().pathString == "/World/Forklift":
        continue
    if prim.GetName().lower().endswith("fork") and prim.IsA(UsdGeom.Xformable):
        fork_prim = XFormPrim(prim.GetPath().pathString)
        break

# ------------------------------------------ choreography (proven v1 table) --
WAYPOINTS = [
    (0.0,  0.0,  3.5,    0, 0.05),
    (3.0, -0.3, -1.0,    0, 0.05),
    (5.0, -0.3, -1.0,  -90, 0.05),
    (7.0, -1.4, -1.0,  -90, 0.05),
    (9.0, -1.4, -1.0,  -90, 0.90),
    (11.0, -0.3, -1.0,  -90, 0.90),
    (14.0,  0.0,  1.0,   90, 0.90),
    (16.0,  1.7,  1.0,   90, 0.90),
    (18.0,  1.7,  1.0,   90, 0.90),
    (20.0,  1.7,  1.0,   90, 0.15),
    (22.0,  0.5,  1.0,   90, 0.05),
    (30.0, -0.5,  3.0,   30, 0.05),   # slow park loop while the orbit finishes
]
CARRY_START, CARRY_END = 7.5, 20.0
FORK_FORWARD = 1.5


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
    if CARRY_START <= t <= CARRY_END:
        bx = x + FORK_FORWARD * np.sin(yaw_r)
        by = y - FORK_FORWARD * np.cos(yaw_r)
        hero.set_world_pose(position=np.array([bx, by, fh]),
                            orientation=euler_angles_to_quat(np.array([0.0, 0.0, yaw_r])))


# --------------------------------------------------- camera + annotators ----
camera = Camera(prim_path="/World/Cam", position=np.array([4.4, -4.4, 5.0]),
                frequency=FPS, resolution=(PANEL_W, PANEL_H))
camera.initialize()
world.reset()
camera.initialize()

usd_cam = UsdGeom.Camera(stage.GetPrimAtPath("/World/Cam"))
usd_cam.GetFocalLengthAttr().Set(usd_cam.GetFocalLengthAttr().Get() * 0.4)
usd_cam.GetFStopAttr().Set(0.0)

# One render product, three annotators reading the SAME frame.
render_product = rep.create.render_product("/World/Cam", (PANEL_W, PANEL_H))
annot_rgb = rep.AnnotatorRegistry.get_annotator("rgb")
annot_depth = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
annot_seg = rep.AnnotatorRegistry.get_annotator("semantic_segmentation",
                                                init_params={"colorize": True})
for a in (annot_rgb, annot_depth, annot_seg):
    a.attach(render_product)

LOOK = np.array([0.0, 0.0, 0.8])


def place_camera(t):
    """Slow orbit around the room, elevated three-quarter view."""
    theta = np.radians(225) + 2 * np.pi * (t / (NUM_FRAMES / FPS)) * 0.6
    pos = LOOK + np.array([6.2 * np.cos(theta), 6.2 * np.sin(theta), 4.6])
    d = LOOK - pos
    yaw = np.arctan2(d[1], d[0])
    pitch = np.arctan2(-d[2], np.linalg.norm(d[:2]))
    camera.set_world_pose(position=pos, orientation=euler_angles_to_quat(np.array([0.0, pitch, yaw])))


def compose_panels():
    """3 panels from 3 annotators: this is the whole point of the demo."""
    rgb = annot_rgb.get_data()
    depth = annot_depth.get_data()
    seg = annot_seg.get_data()
    if rgb is None or rgb.size == 0:
        return None
    p_rgb = cv2.cvtColor(rgb[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR)
    # depth: float meters -> clip, normalize, colormap. inf = sky/nothing hit.
    d = np.nan_to_num(depth, nan=DEPTH_MAX, posinf=DEPTH_MAX)
    d = np.clip(d, 0, DEPTH_MAX) / DEPTH_MAX
    p_depth = cv2.applyColorMap(((1.0 - d) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    # segmentation: already colorized RGBA by the annotator
    seg_img = seg["data"] if isinstance(seg, dict) else seg
    p_seg = cv2.cvtColor(np.asarray(seg_img)[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR)
    for img, txt in [(p_rgb, "RGB"), (p_depth, "DEPTH"), (p_seg, "SEMANTICS")]:
        cv2.putText(img, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return np.hstack([p_rgb, p_depth, p_seg])


os.makedirs(FRAMES_DIR, exist_ok=True)
for f in os.listdir(FRAMES_DIR):
    os.remove(os.path.join(FRAMES_DIR, f))

print("Warming up renderer...")
apply_frame(0.0)
place_camera(0.0)
for _ in range(40):
    world.render()

if SMOKE:
    for t in [0.0, 9.0, 16.0, 25.0]:
        apply_frame(t)
        place_camera(t)
        for _ in range(15):
            world.render()
        frame = compose_panels()
        if frame is not None:
            cv2.imwrite(f"/workspace/smoke_sensor_t{int(t*10):03d}.jpg", frame)
            print(f"SMOKE t={t}: saved shape={frame.shape}")
        else:
            print(f"SMOKE t={t}: EMPTY")
else:
    saved = 0
    for i in range(NUM_FRAMES):
        t = i / FPS
        apply_frame(t)
        place_camera(t)
        world.render()
        frame = compose_panels()
        if frame is not None:
            cv2.imwrite(os.path.join(FRAMES_DIR, f"frame_{i:04d}.jpg"), frame)
            saved += 1
        if (i + 1) % 50 == 0:
            print(f"Recorded {i + 1}/{NUM_FRAMES} frames")
    print(f"Done. Saved {saved}/{NUM_FRAMES} frames to {FRAMES_DIR}")

simulation_app.close()
