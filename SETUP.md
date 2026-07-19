# Setup & Running Guide

Everything needed to run these demos yourself, with zero assistance.

## The environment

- **Where things run:** a RunPod GPU pod (RTX 4000 Ada, 20 GB VRAM, Ubuntu 22.04).
- **Isaac Sim 5.1.0**, installed via pip in a Python 3.11 venv at `/workspace/isaac_env`.
- `/workspace` is the pod's **persistent volume** — it survives stop/start. Everything else (apt packages, `/root`, caches) is wiped every time the pod stops. This one fact explains most "it worked yesterday" failures.

## After EVERY pod start (mandatory)

```bash
bash /workspace/setup.sh
# or manually:
apt-get update && apt-get install -y xvfb ffmpeg libglu1-mesa libegl1
```

> **Why:** system packages don't persist. Missing `xvfb`/`ffmpeg` fails loudly.
> Missing **`libglu1-mesa` fails silently**: Isaac Sim starts fine, but the RTX
> material system (MDL/neuray) can't load, every shader build fails, and the
> camera returns empty frames with no error. This cost a full render cycle to
> diagnose — install it first, always.

Also note: the pod's SSH **IP/port change on every restart** — get them from the
RunPod console. The first Isaac launch after a restart re-downloads the Kit
extension cache (~5–10 min, one time), because it lives outside `/workspace`.

## The launch recipe (every demo uses it)

```bash
export OMNI_KIT_ACCEPT_EULA=yes                       # or Isaac blocks on a prompt
xvfb-run -a -s "-screen 0 1280x720x24" \              # fake X display for headless
    /workspace/isaac_env/bin/python3 -u <script>.py   # -u = live log output
ffmpeg -framerate 20 -i <frames_dir>/frame_%04d.jpg \
       -c:v libx264 -pix_fmt yuv420p <output>.mp4
```

For long renders, detach and poll:

```bash
nohup bash -c '<the command above> > /workspace/job.log 2>&1; echo EXIT_$? >> /workspace/job.log' &
grep EXIT_ /workspace/job.log   # job is done when this prints
```

## Per-demo commands

Each script supports `SMOKE=1` (render a few test stills instead of the full
video — **always run this first**) and `NUM_FRAMES=<n>` to override length.

| Demo | Script | Full render | Output |
|---|---|---|---|
| City orbit | `simple_city_car.py` | ~10 min | 10 s, 200 frames |
| Warehouse (kinematic) | `warehouse/warehouse_forklift.py` | ~12 min | 24 s, 480 frames |
| Warehouse (physics) | `physics/warehouse_physics.py` | ~15 min | 29 s, 580 frames |
| Sensor rig | `sensors/sensor_rig.py` | ~15 min | 30 s, 3-panel 1920x360 |
| City delivery | `city_delivery/city_delivery.py` | ~30 min | 60 s, 1200 frames |
| RL Jetbot | `rl/rl_jetbot_cem.py` | ~20 min (train+render) | ~45 s + learning curve |

Example (sensor rig, smoke first):

```bash
cd /workspace
export OMNI_KIT_ACCEPT_EULA=yes
SMOKE=1 xvfb-run -a -s "-screen 0 1280x720x24" /workspace/isaac_env/bin/python3 -u sensor_rig.py
# inspect /workspace/smoke_sensor_*.jpg, then:
xvfb-run -a -s "-screen 0 1280x720x24" /workspace/isaac_env/bin/python3 -u sensor_rig.py
ffmpeg -framerate 20 -i /workspace/sensor_frames/frame_%04d.jpg -c:v libx264 -pix_fmt yuv420p sensor_rig.mp4
```

## Getting files on/off the pod

```bash
scp -P <PORT> -i ~/.ssh/id_ed25519 script.py root@<IP>:/workspace/      # upload
scp -P <PORT> -i ~/.ssh/id_ed25519 root@<IP>:/workspace/out.mp4 .       # download
```

## Verifying assets before using them (do this, seriously)

`add_reference_to_stage` does **not** error on nonexistent paths — the prim just
stays empty. Check the public asset bucket first, no pod needed:

```bash
curl -s "https://omniverse-content-production.s3-us-west-2.amazonaws.com/?list-type=2&delimiter=/&prefix=Assets/Isaac/5.1/Isaac/Robots/NVIDIA/" | grep -oE '<Prefix>[^<]+'
```

## Stopping the pod (billing!)

GPU billing runs while the pod is up. From inside the pod:

```bash
export $(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(API_KEY|POD_ID)=' | xargs)
runpodctl stop pod $RUNPOD_POD_ID
```

A stopped pod still bills a small storage fee for `/workspace`. Terminate from
the console to stop that too — but that **deletes `/workspace` permanently**.
