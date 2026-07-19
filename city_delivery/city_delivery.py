"""City delivery sim — Isaac Sim 5.1, headless. 60 seconds, three robots.

A delivery vignette staged on a verified road section of the Rivermark city:
* LEATHERBACK (car-scale) runs two delivery loops: picks a cardboard box up
  at the depot, drives it across the road section, drops it, returns for the
  next one.
* CARTER patrols the far lane back and forth.
* JETBOT (scaled up) loops a small rectangle near the depot.

WHAT THIS TEACHES:
* Multi-actor kinematic choreography: every robot is driven by the same
  waypoint-table interpolator, just with its own table. Scaling from one
  actor to N is data, not new code.
* Scripted CAMERA CUTS: the camera is itself an actor with a schedule —
  follow-cam segments compute pose from the tracked robot's pose each frame,
  static segments are fixed waypoints. Film language, implemented in 20 lines.
* Event scripting: pickup/dropoff are time-windowed pose-follows (the proven
  attach trick); outside their windows boxes keep their last pose.
* Assumption-checking: each asset's forward axis and authored scale are
  verified in smoke stills before the full render (Leatherback is authored
  in centimeters — scale 0.05 makes it a ~2.1 m car).

Run (on the pod):
  export OMNI_KIT_ACCEPT_EULA=yes
  xvfb-run -a -s "-screen 0 1280x720x24" \
      /workspace/isaac_env/bin/python3 -u city_delivery.py
  ffmpeg -framerate 20 -i /workspace/city_frames/frame_%04d.jpg \
         -c:v libx264 -pix_fmt yuv420p city_delivery.mp4
"""
import os
import numpy as np
import cv2

print("Starting Isaac Sim (headless)...")
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

from omni.isaac.core import World
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.sensor import Camera
import omni
from pxr import UsdGeom, UsdLux, Gf

FRAMES_DIR = "/workspace/city_frames"
FPS = 20
DURATION = 60.0
NUM_FRAMES = int(os.environ.get("NUM_FRAMES", str(int(DURATION * FPS))))
SMOKE = os.environ.get("SMOKE") == "1"

# Road-surface anchor found by probing Rivermark's roadmark tile bboxes
# (see the city demo). All route coordinates below are offsets from ROAD.
ROAD = np.array([268.9, 46.8, 6.15])

world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()
assets_root = get_assets_root_path()

print("Loading Rivermark city (takes a few minutes)...")
add_reference_to_stage(usd_path=assets_root + "/Isaac/Environments/Outdoor/Rivermark/rivermark.usd",
                       prim_path="/World/City")

sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
sun.CreateIntensityAttr(2500)
UsdGeom.XformCommonAPI(sun.GetPrim()).SetRotate(Gf.Vec3f(-50, 0, 40))
dome = UsdLux.DomeLight.Define(stage, "/World/Dome")
dome.CreateIntensityAttr(600)


def place(usd_path, prim_path, offset, yaw_deg=0.0, scale=None):
    add_reference_to_stage(usd_path=assets_root + usd_path, prim_path=prim_path)
    xf = XFormPrim(prim_path)
    xf.set_world_pose(position=ROAD + np.array(offset, dtype=float),
                      orientation=euler_angles_to_quat(np.array([0.0, 0.0, np.radians(yaw_deg)])))
    if scale is not None:
        xf.set_local_scale(np.array([scale] * 3, dtype=float))
    return xf


# ------------------------------------------------------------------ actors --
# Verified assets; scales normalize authored units to world size
van = place("/Isaac/Robots/NVIDIA/Leatherback/leatherback.usd", "/World/Van",
            [-12.0, 6.0, 0.05], scale=0.05)              # cm-authored -> 2.1 m car
carter = place("/Isaac/Robots/NVIDIA/Carter/carter_v1.usd", "/World/Carter",
               [14.0, -7.0, 0.1], scale=1.5)
jetbot = place("/Isaac/Robots/NVIDIA/Jetbot/jetbot.usd", "/World/Jetbot",
               [-14.0, 3.0, 0.05], scale=4.0)

WHBOX = "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxA_01.usd"
DEPOT = np.array([-12.0, 7.5, 0.0])
DROP = np.array([12.0, -4.0, 0.0])
box1 = place(WHBOX, "/World/Box1", DEPOT, yaw_deg=10, scale=0.8)
box2 = place(WHBOX, "/World/Box2", DEPOT + [0.0, 0.9, 0.0], yaw_deg=-15, scale=0.8)
# depot dressing: a pallet and one more box that stays put
place("/Isaac/Props/Pallet/pallet.usd", "/World/DepotPallet", DEPOT + [1.2, 0.4, 0.0])
place(WHBOX, "/World/DepotBox", DEPOT + [1.2, 0.4, 0.15], yaw_deg=35, scale=0.8)

# Per-asset yaw offset: which way the asset faces at yaw=0 differs per model.
# Verified via smoke stills; adjust here if a robot drives sideways.
YAW_OFFSET = {"van": 0.0, "carter": 0.0, "jetbot": 0.0}

# --------------------------------------------------------------- routes -----
# (time, x_off, y_off, yaw_deg). Same interpolator as every other demo.
VAN_ROUTE = [
    (0.0, -12.0, 6.0,   0),
    (3.0, -12.0, 6.0,   0),     # loading pause (box1 attaches at t=2)
    (10.0, -2.0, 2.0, -20),     # pull out, cross the road section
    (17.0, 10.0, -3.0, -25),
    (20.0, 12.0, -4.0, -30),    # arrive at drop zone (box1 detaches t=21)
    (23.0, 12.0, -4.0, -30),    # unloading pause
    (30.0,  2.0, 3.0, 160),     # loop back to depot
    (36.0, -12.0, 6.0, 185),
    (39.0, -12.0, 6.0, 180),    # load box2 (attaches t=38)
    (47.0, -1.0, 1.5, -15),     # second run
    (55.0, 11.5, -3.5, -28),
    (58.0, 12.0, -4.0, -30),    # deliver box2 (detaches t=59)
    (60.0, 12.0, -4.0, -30),
]
CARTER_ROUTE = [
    (0.0, 14.0, -7.0, 180), (12.0, -14.0, -7.0, 180), (14.0, -14.0, -7.0, 0),
    (26.0, 14.0, -7.0, 0), (28.0, 14.0, -7.0, 180), (40.0, -14.0, -7.0, 180),
    (42.0, -14.0, -7.0, 0), (54.0, 14.0, -7.0, 0), (60.0, 14.0, -7.0, 0),
]
JETBOT_ROUTE = [
    (0.0, -14.0, 3.0, 0), (8.0, -8.0, 3.0, 0), (10.0, -8.0, 3.0, 90),
    (14.0, -8.0, 8.0, 90), (16.0, -8.0, 8.0, 180), (24.0, -14.0, 8.0, 180),
    (26.0, -14.0, 8.0, 270), (30.0, -14.0, 3.0, 270), (32.0, -14.0, 3.0, 0),
    (40.0, -8.0, 3.0, 0), (42.0, -8.0, 3.0, 90), (46.0, -8.0, 8.0, 90),
    (48.0, -8.0, 8.0, 180), (56.0, -14.0, 8.0, 180), (60.0, -14.0, 8.0, 180),
]

# box carry windows: (box, attach_t, detach_t, rest_pose_after)
CARRIES = [
    ("box1", 2.0, 21.0, DROP + [0.0, 0.0, 0.0]),
    ("box2", 38.0, 59.0, DROP + [0.0, 1.0, 0.0]),
]
CARRY_HEIGHT = 0.95   # box rides on the van's rear deck


def smoothstep(u):
    return u * u * (3 - 2 * u)


def route_pose(route, t):
    if t <= route[0][0]:
        _, x, y, yaw = route[0]
        return x, y, yaw
    for (t0, x0, y0, w0), (t1, x1, y1, w1) in zip(route, route[1:]):
        if t0 <= t <= t1:
            u = smoothstep((t - t0) / (t1 - t0))
            return x0 + u * (x1 - x0), y0 + u * (y1 - y0), w0 + u * (w1 - w0)
    _, x, y, yaw = route[-1]
    return x, y, yaw


def set_actor(xf, route, t, z, yaw_off):
    x, y, yaw = route_pose(route, t)
    xf.set_world_pose(position=ROAD + np.array([x, y, z]),
                      orientation=euler_angles_to_quat(np.array([0.0, 0.0, np.radians(yaw + yaw_off)])))
    return x, y, yaw


def apply_frame(t):
    vx, vy, vyaw = set_actor(van, VAN_ROUTE, t, 0.05, YAW_OFFSET["van"])
    set_actor(carter, CARTER_ROUTE, t, 0.1, YAW_OFFSET["carter"])
    set_actor(jetbot, JETBOT_ROUTE, t, 0.05, YAW_OFFSET["jetbot"])
    for name, t_on, t_off, rest in CARRIES:
        box = box1 if name == "box1" else box2
        if t_on <= t <= t_off:
            # ride slightly behind the van's center, on its deck
            yr = np.radians(vyaw)
            pos = ROAD + np.array([vx - 0.55 * np.cos(yr), vy - 0.55 * np.sin(yr), CARRY_HEIGHT])
            box.set_world_pose(position=pos,
                               orientation=euler_angles_to_quat(np.array([0.0, 0.0, yr])))
        elif t > t_off:
            box.set_world_pose(position=ROAD + rest + np.array([0.0, 0.0, 0.0]),
                               orientation=euler_angles_to_quat(np.array([0.0, 0.0, 0.3])))
    return vx, vy, vyaw


# ---------------------------------------------------------------- cameras ---
camera = Camera(prim_path="/World/Cam", position=ROAD + np.array([0, 0, 10]),
                frequency=FPS, resolution=(1280, 720))
camera.initialize()
world.reset()
camera.initialize()
usd_cam = UsdGeom.Camera(stage.GetPrimAtPath("/World/Cam"))
usd_cam.GetFocalLengthAttr().Set(usd_cam.GetFocalLengthAttr().Get() * 0.45)
usd_cam.GetFStopAttr().Set(0.0)


def look_from(pos, target):
    d = target - pos
    yaw = np.arctan2(d[1], d[0])
    pitch = np.arctan2(-d[2], np.linalg.norm(d[:2]))
    camera.set_world_pose(position=pos,
                          orientation=euler_angles_to_quat(np.array([0.0, pitch, yaw])))


def place_camera(t, vx, vy, vyaw):
    van_pos = ROAD + np.array([vx, vy, 0.8])
    yr = np.radians(vyaw)
    if t < 22.0:      # follow-cam behind the van
        pos = van_pos + np.array([-6.5 * np.cos(yr), -6.5 * np.sin(yr), 2.8])
        look_from(pos, van_pos)
    elif t < 34.0:    # static wide: the whole intersection
        look_from(ROAD + np.array([-4.0, -14.0, 9.0]), ROAD + np.array([0, 0, 1.0]))
    elif t < 50.0:    # low side-tracking shot
        pos = van_pos + np.array([-5.5 * np.sin(yr) * -1.0, 5.5 * np.cos(yr) * -1.0, 1.2])
        look_from(pos, van_pos)
    else:             # high slow orbit finale
        theta = np.radians(200) + (t - 50.0) * 0.12
        look_from(ROAD + np.array([16 * np.cos(theta), 16 * np.sin(theta), 11.0]),
                  ROAD + np.array([4.0, -2.0, 0.5]))


os.makedirs(FRAMES_DIR, exist_ok=True)
for f in os.listdir(FRAMES_DIR):
    os.remove(os.path.join(FRAMES_DIR, f))

print("Warming up renderer...")
vx, vy, vyaw = apply_frame(0.0)
place_camera(0.0, vx, vy, vyaw)
for _ in range(40):
    world.render()


def capture(path):
    img = camera.get_rgba()
    if img is not None and img.size > 0:
        cv2.imwrite(path, cv2.cvtColor(img[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR))
        return True
    return False


if SMOKE:
    for t in [0.0, 6.0, 15.0, 21.0, 28.0, 42.0, 57.0]:
        vx, vy, vyaw = apply_frame(t)
        place_camera(t, vx, vy, vyaw)
        for _ in range(15):
            world.render()
        ok = capture(f"/workspace/smoke_city_t{int(t*10):03d}.jpg")
        print(f"SMOKE t={t}: {'saved' if ok else 'EMPTY'} van=({vx:.1f},{vy:.1f},yaw {vyaw:.0f})")
else:
    saved = 0
    for i in range(NUM_FRAMES):
        t = i / FPS
        vx, vy, vyaw = apply_frame(t)
        place_camera(t, vx, vy, vyaw)
        world.render()
        if capture(os.path.join(FRAMES_DIR, f"frame_{i:04d}.jpg")):
            saved += 1
        if (i + 1) % 100 == 0:
            print(f"Recorded {i + 1}/{NUM_FRAMES} frames")
    print(f"Done. Saved {saved}/{NUM_FRAMES} frames to {FRAMES_DIR}")

simulation_app.close()
