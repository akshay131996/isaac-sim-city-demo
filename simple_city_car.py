"""Simple cityscape + car view: orbit camera around a sedan in the Rivermark environment."""
import os
import numpy as np
import cv2

print("Starting Isaac Sim (headless)...")
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

from omni.isaac.core import World
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.sensor import Camera
import omni
from pxr import UsdGeom, UsdLux, Gf

FRAMES_DIR = "/workspace/simple_city_frames"
NUM_FRAMES = int(os.environ.get("NUM_FRAMES", "200"))   # 10 seconds at 20 fps
WARMUP_STEPS = 40

world = World(stage_units_in_meters=1.0)
assets_root = get_assets_root_path()
if assets_root is None:
    print("ERROR: assets root not found")
    simulation_app.close()
    raise SystemExit(1)

print("Loading Rivermark city environment...")
add_reference_to_stage(usd_path=assets_root + "/Isaac/Environments/Outdoor/Rivermark/rivermark.usd",
                       prim_path="/World/City")

# road surface found by probing bounding boxes of flat roadmark tiles
SPOT = np.array([268.9, 46.8, 6.15])

print("Spawning car (Leatherback)...")
add_reference_to_stage(usd_path=assets_root + "/Isaac/Robots/NVIDIA/Leatherback/leatherback.usd",
                       prim_path="/World/Car")
stage = omni.usd.get_context().get_stage()
# XformCommonAPI silently no-ops on prims with incompatible xform stacks;
# XFormPrim handles any op order.
from omni.isaac.core.prims import XFormPrim
car = XFormPrim("/World/Car")
car.set_world_pose(position=np.array([SPOT[0], SPOT[1], SPOT[2] + 0.05]))
# The Leatherback USD is authored in centimeters (0.01) and is RC-scale
# (~0.4 m); 0.05 lands at full-car size (~2.1 m). Safe to scale because we
# never step physics — the render loop below only updates the renderer.
car.set_local_scale(np.array([0.05, 0.05, 0.05]))

# extra lighting so the shot isn't dark
sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
sun.CreateIntensityAttr(2500)
UsdGeom.XformCommonAPI(sun.GetPrim()).SetRotate(Gf.Vec3f(-50, 0, 40))
dome = UsdLux.DomeLight.Define(stage, "/World/Dome")
dome.CreateIntensityAttr(600)

camera = Camera(prim_path="/World/OrbitCam",
                position=SPOT + np.array([5.0, 0.0, 2.0]),
                frequency=20,
                resolution=(1280, 720))
camera.initialize()
world.reset()
camera.initialize()

look_at = SPOT + np.array([0.0, 0.0, 0.6])  # roughly the car body
radius, cam_height = 6.0, SPOT[2] + 2.0

def place_camera(theta):
    pos = np.array([look_at[0] + radius * np.cos(theta),
                    look_at[1] + radius * np.sin(theta),
                    cam_height])
    d = look_at - pos
    yaw = np.arctan2(d[1], d[0])
    pitch = np.arctan2(-d[2], np.linalg.norm(d[:2]))
    quat = euler_angles_to_quat(np.array([0.0, pitch, yaw]))
    camera.set_world_pose(position=pos, orientation=quat)

os.makedirs(FRAMES_DIR, exist_ok=True)
for f in os.listdir(FRAMES_DIR):
    os.remove(os.path.join(FRAMES_DIR, f))

from pxr import Usd
bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"]).ComputeWorldBound(
    stage.GetPrimAtPath("/World/Car")).ComputeAlignedRange()
print("CAR BBOX:", "EMPTY" if bbox.IsEmpty() else (tuple(bbox.GetMin()), tuple(bbox.GetMax())))

print("Warming up renderer...")
place_camera(0.0)
for _ in range(WARMUP_STEPS):
    world.render()

print("Recording frames...")
saved = 0
for i in range(NUM_FRAMES):
    place_camera(2.0 * np.pi * i / NUM_FRAMES)
    world.render()
    img = camera.get_rgba()
    if img is not None and img.shape[0] > 0:
        bgr = cv2.cvtColor(img[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(FRAMES_DIR, f"frame_{i:04d}.jpg"), bgr)
        saved += 1
    else:
        print(f"Warning: empty frame at {i}")
    if (i + 1) % 25 == 0:
        print(f"Recorded {i + 1}/{NUM_FRAMES} frames")

print(f"Done. Saved {saved}/{NUM_FRAMES} frames to {FRAMES_DIR}")
simulation_app.close()
