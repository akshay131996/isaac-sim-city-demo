"""RL learning demo — Jetbot learns goal-reaching, Isaac Sim 5.1, headless.

A differential-drive Jetbot learns to drive to a goal marker using the
CROSS-ENTROPY METHOD (CEM) — evolutionary policy search, one of the
simplest true reinforcement-learning algorithms. No Isaac Lab, no PyTorch:
the entire learning loop is ~40 lines of numpy on top of the physics sim.

WHAT THIS TEACHES:
* The RL loop anatomy: observation -> policy -> action -> physics step ->
  reward, repeated over episodes. Identical structure to PPO/SAC pipelines,
  minus the gradient machinery.
* OBSERVATIONS are engineered: the policy sees [sin(heading_err),
  cos(heading_err), normalized_distance] — 3 numbers, in the ROBOT's frame.
  Frame choice is what makes the task learnable by 8 parameters.
* POLICY: linear map obs -> two wheel velocities. W (2x3) + b (2) = 8 params.
* CEM: sample 16 parameter vectors from a Gaussian, roll each out for one
  episode, keep the 4 best ("elites"), refit the Gaussian to them, repeat.
  Selection pressure does the learning — no gradients anywhere.
* Physics-in-the-loop training: every episode is real simulation (wheel
  friction, inertia); resets must restore state exactly (world.reset()).
* Why training is HEADLESS: no rendering during rollouts makes sim steps
  ~10x faster. Rendering happens only for the demo video afterwards.

Outputs:
  /workspace/rl_frames/*.jpg   video frames: pre/mid/post-training episodes
                               with captions, plus a learning-curve panel
  /workspace/rl_policy.npz     trained parameters + per-iteration rewards
  stdout                       the learning curve, iteration by iteration

Run (on the pod):
  export OMNI_KIT_ACCEPT_EULA=yes
  xvfb-run -a -s "-screen 0 1280x720x24" \
      /workspace/isaac_env/bin/python3 -u rl_jetbot_cem.py
  ffmpeg -framerate 20 -i /workspace/rl_frames/frame_%05d.jpg \
         -c:v libx264 -pix_fmt yuv420p rl_jetbot.mp4
"""
import os
import numpy as np
import cv2

print("Starting Isaac Sim (headless)...")
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

from omni.isaac.core import World
from omni.isaac.core.objects import VisualCuboid
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.sensor import Camera
import omni
from pxr import UsdGeom, UsdLux, Gf

FRAMES_DIR = "/workspace/rl_frames"
FPS = 20
PHYS_DT = 1.0 / 60.0
EP_STEPS = 300                # 5 s episodes at 60 Hz — goals must be
                              # REACHABLE within an episode or the success
                              # bonus never fires and there is no signal
POP, ELITES, ITERS = 16, 4, 10
WHEEL_SPEED = 15.0            # action scale, rad/s
SMOKE = os.environ.get("SMOKE") == "1"
rng = np.random.default_rng(7)

world = World(stage_units_in_meters=1.0, physics_dt=PHYS_DT, rendering_dt=1.0 / FPS)
stage = omni.usd.get_context().get_stage()
assets_root = get_assets_root_path()

world.scene.add_default_ground_plane()
add_reference_to_stage(usd_path=assets_root + "/Isaac/Robots/NVIDIA/Jetbot/jetbot.usd",
                       prim_path="/World/Jetbot")
robot = Articulation(prim_path="/World/Jetbot", name="jetbot")
world.scene.add(robot)
robot_xf = XFormPrim("/World/Jetbot")

goal_vis = VisualCuboid(prim_path="/World/Goal", position=np.array([1.5, 0.0, 0.05]),
                        scale=np.array([0.15, 0.15, 0.10]), color=np.array([0.1, 0.9, 0.2]))

dome = UsdLux.DomeLight.Define(stage, "/World/Dome")
dome.CreateIntensityAttr(1200)
sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
sun.CreateIntensityAttr(1800)
UsdGeom.XformCommonAPI(sun.GetPrim()).SetRotate(Gf.Vec3f(-50, 0, 30))

camera = Camera(prim_path="/World/Cam", position=np.array([2.6, -2.6, 2.2]),
                frequency=FPS, resolution=(1280, 720))
camera.initialize()
world.reset()
camera.initialize()
usd_cam = UsdGeom.Camera(stage.GetPrimAtPath("/World/Cam"))
usd_cam.GetFocalLengthAttr().Set(usd_cam.GetFocalLengthAttr().Get() * 0.5)
usd_cam.GetFStopAttr().Set(0.0)

# find the wheel joints by name — never hardcode joint indices
dof_names = robot.dof_names
print("DOF NAMES:", dof_names)
left_idx = [i for i, n in enumerate(dof_names) if "left" in n.lower()][0]
right_idx = [i for i, n in enumerate(dof_names) if "right" in n.lower()][0]


def aim_camera(target):
    pos = np.array([2.6, -2.6, 2.2])
    d = target - pos
    yaw = np.arctan2(d[1], d[0])
    pitch = np.arctan2(-d[2], np.linalg.norm(d[:2]))
    camera.set_world_pose(position=pos, orientation=euler_angles_to_quat(np.array([0.0, pitch, yaw])))


aim_camera(np.array([0.7, 0.0, 0.1]))


def get_obs(goal):
    """Observation in the ROBOT frame — this is the feature engineering that
    makes an 8-parameter policy sufficient.

    NOTE: pose comes from the ARTICULATION (physics view), not XFormPrim.
    With Fabric enabled, simulated poses never write back to USD, so an
    XFormPrim observer reads the frozen spawn pose forever — the policy
    goes blind while the robot drives (v2 bug, and LEARNINGS.md #17)."""
    pos, quat = robot.get_world_pose()
    yaw = quat_to_euler_angles(quat)[2]
    to_goal = goal[:2] - pos[:2]
    dist = float(np.linalg.norm(to_goal))
    err = np.arctan2(to_goal[1], to_goal[0]) - yaw
    err = np.arctan2(np.sin(err), np.cos(err))   # wrap to [-pi, pi]
    return np.array([np.sin(err), np.cos(err), min(dist, 3.0) / 3.0]), dist


def policy_action(params, obs):
    W = params[:6].reshape(2, 3)
    b = params[6:8]
    v = np.tanh(W @ obs + b) * WHEEL_SPEED
    return v  # [left, right] wheel velocity


def run_episode(params, goal, record=None, caption=""):
    """One rollout. record: list to append video frames to (None = headless)."""
    world.reset()
    # Command wheels through the articulation CONTROLLER in velocity mode.
    # v1 bug: set_joint_velocities writes joint STATE, which the joint
    # DRIVES immediately override — the robot never moved and every policy
    # scored identically. Drives are the motors; talk to the motors.
    controller = robot.get_articulation_controller()
    controller.switch_control_mode("velocity")
    goal_vis.set_world_pose(position=np.array([goal[0], goal[1], 0.05]))
    start_obs, start_dist = get_obs(goal)
    dist = start_dist
    for step in range(EP_STEPS):
        obs, dist = get_obs(goal)
        v = policy_action(params, obs)
        controller.apply_action(ArticulationAction(
            joint_velocities=np.array(v), joint_indices=np.array([left_idx, right_idx])))
        render = record is not None and step % 3 == 0   # 60 Hz physics -> 20 fps video
        world.step(render=render)
        if render:
            img = camera.get_rgba()
            if img is not None and img.size > 0:
                f = cv2.cvtColor(img[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR)
                cv2.putText(f, caption, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.putText(f, f"distance to goal: {dist:.2f} m", (16, 76),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                record.append(f)
        if dist < 0.25:
            break
    # reward: progress toward goal + big bonus for arrival (time-discounted)
    reached = dist < 0.25
    return (start_dist - dist) + (3.0 if reached else 0.0) - 0.002 * step, reached


def sample_goal():
    ang = rng.uniform(-np.pi, np.pi)
    r = rng.uniform(0.8, 1.6)   # reachable within one episode (see EP_STEPS)
    return np.array([r * np.cos(ang), r * np.sin(ang)])


# ------------------------------------------------------------- training -----
mean = np.zeros(8)
std = np.ones(8) * 1.0
history = []          # (iteration, mean_reward, best_reward, success_rate)
first_iter_best = None

print(f"Training: CEM pop={POP} elites={ELITES} iters={ITERS} ep={EP_STEPS} steps")
eval_goals = [sample_goal() for _ in range(3)]   # fixed goals: fair comparison

for it in range(ITERS):
    candidates = rng.normal(mean, std, size=(POP, 8))
    rewards, successes = [], 0
    for c in candidates:
        r, ok = run_episode(c, sample_goal())
        rewards.append(r)
        successes += int(ok)
    rewards = np.array(rewards)
    elite_idx = rewards.argsort()[-ELITES:]
    elites = candidates[elite_idx]
    mean = elites.mean(axis=0)
    std = elites.std(axis=0) + 0.05        # noise floor keeps exploring
    if it == 0:
        first_iter_best = candidates[rewards.argmax()].copy()
    history.append((it, float(rewards.mean()), float(rewards.max()), successes / POP))
    print(f"ITER {it}: mean_r={rewards.mean():+.2f} best_r={rewards.max():+.2f} "
          f"success={successes}/{POP}")

np.savez("/workspace/rl_policy.npz", mean=mean, history=np.array(history))
print("Saved policy to /workspace/rl_policy.npz")

if SMOKE:
    print("SMOKE mode: training verified, skipping video render")
    simulation_app.close()
    raise SystemExit(0)

# ------------------------------------------------------- render the story ---
os.makedirs(FRAMES_DIR, exist_ok=True)
for f in os.listdir(FRAMES_DIR):
    os.remove(os.path.join(FRAMES_DIR, f))

frames = []


def title_card(lines, seconds=2.0):
    for _ in range(int(seconds * FPS)):
        img = np.zeros((720, 1280, 3), dtype=np.uint8)
        for j, line in enumerate(lines):
            cv2.putText(img, line, (80, 300 + 60 * j), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (255, 255, 255), 2)
        frames.append(img)


title_card(["RL demo: Jetbot learns goal-reaching", "Cross-Entropy Method, 8 parameters"], 2.5)
title_card(["BEFORE training", "(best of 16 random policies, iteration 0)"], 2.0)
for g in eval_goals[:2]:
    run_episode(first_iter_best, g, record=frames, caption="BEFORE training (iteration 0)")
title_card(["AFTER training", f"({ITERS} CEM iterations, ~{ITERS * POP} episodes)"], 2.0)
for g in eval_goals:
    run_episode(mean, g, record=frames, caption=f"AFTER training (iteration {ITERS})")

# learning-curve panel drawn with cv2 (no plotting libs needed)
curve = np.full((720, 1280, 3), 20, dtype=np.uint8)
means = [h[1] for h in history]
lo, hi = min(means) - 0.5, max(means) + 0.5
pts = []
for i, m in enumerate(means):
    x = int(120 + i * (1040 / max(1, ITERS - 1)))
    y = int(620 - (m - lo) / (hi - lo) * 480)
    pts.append((x, y))
for a, b in zip(pts, pts[1:]):
    cv2.line(curve, a, b, (80, 220, 120), 3)
for (x, y), (_, m, _, sr) in zip(pts, history):
    cv2.circle(curve, (x, y), 6, (255, 255, 255), -1)
cv2.putText(curve, "mean episode reward per CEM iteration", (120, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
cv2.putText(curve, f"final success rate: {history[-1][3] * 100:.0f}%", (120, 130),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 255, 150), 2)
for _ in range(int(4.0 * FPS)):
    frames.append(curve)

for i, f in enumerate(frames):
    cv2.imwrite(os.path.join(FRAMES_DIR, f"frame_{i:05d}.jpg"), f)
print(f"Done. Wrote {len(frames)} video frames ({len(frames) / FPS:.1f} s) to {FRAMES_DIR}")

simulation_app.close()
