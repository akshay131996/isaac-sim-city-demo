# Warehouse XL

A larger, busier variant of the warehouse forklift demo — Isaac Sim 5.1, headless. Separate from the original `warehouse/warehouse_forklift.py`, which is left untouched.

![demo](demo.gif)

*Full-quality video: [warehouse_xl.mp4](warehouse_xl.mp4) (30 s)*

## What's in it

- **20 × 14 m warehouse** — 4× the floor area of the original 10 × 10 room.
- **Two vehicles**: a forklift runs a pickup → carry → drop between the west and east aisles, while an **NVIDIA Leatherback car patrols the open floor**.
- **More assets**: six 2-tier shelving racks stocked with cardboard boxes and KLT bins along both walls, plastic barrels, a staged pallet, and "KEEP CLEAR" floor decals.

## Techniques (what this demo teaches on top of the original)

- **Multi-actor kinematic choreography** — each vehicle has its own waypoint table, driven by the same smoothstep interpolator. Scaling from one actor to N is data, not new code.
- **Auto-heading** — the car's facing is derived from its travel direction each frame (`yaw = atan2(dy, dx)`), so you never hand-tune per-waypoint yaw for it. If a vehicle ever drives sideways, its model's forward axis differs from local +x — correct it with the one-line `CAR_YAW_OFFSET`.
- **Kinematic scenes still pay a PhysX cost.** Even though this demo never steps physics (only `world.render()`), `World.reset()` parses the collision geometry baked into every referenced asset. An over-stuffed first attempt (8 racks + pillars + bottles + 2 vehicles) aborted PhysX during scene init (`SIGABRT` in `physx::Cm::FanoutTask`). The fix was purely a **prop-budget cut** (~20 physics-carrying assets), not a logic change — the asset count has a hard ceiling at scene init regardless of whether you simulate.

## Run

```bash
export OMNI_KIT_ACCEPT_EULA=yes
xvfb-run -a -s "-screen 0 1280x720x24" /workspace/isaac_env/bin/python3 -u warehouse_xl.py
ffmpeg -framerate 20 -i /workspace/warehouse_xl_frames/frame_%04d.jpg -c:v libx264 -pix_fmt yuv420p warehouse_xl.mp4
```

`SMOKE=1` renders a few stills at key beats instead of the full video — always run that first. See the repo's [SETUP.md](../SETUP.md) for the full environment/runbook.
