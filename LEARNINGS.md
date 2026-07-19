# Learnings

Every lesson in this file was earned from an actual failure in these demos —
this is the debugging curriculum, in the order it happened.

## Rendering & environment

1. **Ephemeral vs persistent.** On RunPod only `/workspace` survives a restart.
   apt packages, `/root` caches, and the Omniverse extension cache do not.
   Symptom pattern: "worked yesterday, broken today" after any pod stop.
2. **`libGLU` is load-bearing and fails silently.** Without `libglu1-mesa` the
   MDL material system dies, all RTX shaders fail, and cameras return empty
   frames — while the app runs "successfully". Diagnose renderer problems by
   grepping the log for subsystem errors (`vulkan|rtx|neuray`), not Python
   tracebacks.
3. **Isolate before iterating.** A 2-minute cube-and-camera diagnostic script
   proves the capture pipeline separately from your scene. Never debug both at
   once.
4. **The default camera FOV is ~24°** (telephoto). Overview shots need the
   focal length reduced (we write the USD attribute directly — the Camera
   wrapper's setter segfaulted in this build). Disable depth-of-field with
   f-stop 0.
5. **Smoke-test stills before every full render.** A 5-minute still run caught
   a bug 100% of the times we ran it: bad framing, missing assets, wrong scale,
   wrong axis conventions. Full renders only after stills look right.

## USD & assets

6. **Missing assets are silent.** `add_reference_to_stage` with a bad URL
   creates an empty prim, no error. Verify paths against the S3 bucket with
   `curl` before writing code. (The sedan referenced by older scripts doesn't
   exist in the 5.1 tree at all.)
7. **Print bounding boxes after loading.** One line catches both silent
   failures (empty bbox) and unit mismatches — the Leatherback is authored in
   **centimeters** and loads as an 87 m car until scaled by 0.05.
8. **Never assume an asset's forward axis.** This forklift's forks point along
   local **−y**. Robots differ. One smoke still answers what no documentation
   states.
9. **Name-based prim searches need care**: searching for "fork" matched the
   root prim ("Forklift") and then a body part ("S_ForkliftBody") before
   finding the part we wanted (`S_ForkliftFork`).
10. **`XformCommonAPI` silently no-ops** on prims with pre-existing transform
    stacks. `XFormPrim.set_world_pose` handles any stack. (Fresh prims you
    created yourself are safe with either.)
11. **Scale is applied before orientation** — an unrotated `FixedCuboid` with a
    long scale juts where you don't expect. Pass an orientation; don't assume.

## Physics (the warehouse pickup saga — 13 iterations)

12. **Three body types**: static (scenery), kinematic (animated, pushes
    dynamics, infinite force), dynamic (physics-owned). Choosing wrong is the
    #1 beginner error.
13. **Convex hulls fill concavities.** The single biggest physics lesson here:
    a convex hull around a forklift turns the gap between the fork blades into
    solid matter — an invisible bulldozer blade. Symptom: objects get pushed,
    never penetrated.
14. **Convex *decomposition* can still bridge gaps** on complex single meshes
    (both blades are one mesh; the decomposition walled them together anyway).
15. **The production answer is hand-authored colliders**: invisible primitive
    shapes (cubes) as collision proxies — a deck slab + feet for the pallet,
    two blade boxes for the forks. Exact, fast, controllable. Auto-generated
    collision is a convenience, not a guarantee.
16. **Physics materials must be BOUND, not just defined.** An unbound friction
    material is a silent no-op (recurring theme: USD fails silently).
17. **Fabric means USD lies about simulated poses.** Read dynamic objects
    through `RigidPrim.get_world_pose()` (the physics view); USD bbox caches
    show stale spawn poses.
18. **Instrument, don't guess.** The breakthroughs came from printing world
    bboxes of the fork part (found the tine height formula
    `tine_z = fh + 0.057`), then per-mesh dumps (found the blade lanes), then
    overlap arithmetic. Numbers ended a 6-iteration guessing streak.
19. **Contact offsets create phantom contact** across ~1–2 cm gaps. Design
    clearances of ≥5 cm around PhysX defaults, or tune
    contactOffset/restOffset explicitly.
20. **Timebox subproblems.** The dynamic-pallet slot entry is its own rabbit
    hole; the demo goal was moving the *box*. Restaging the box on dunnage
    blocks (real warehouse practice) delivered the goal and kept the physics
    honest.
21. **Simulation is stateful.** You can't jump to t=19 like in kinematic
    animation — smoke tests must fast-forward *through* every physics step.
    That's the fundamental difference between animation and simulation.

## RL (Jetbot CEM)

22. **Observation design is most of the problem.** Goal direction expressed in
    the robot's frame ([sin/cos of heading error, distance]) makes the task
    solvable by 8 linear parameters. The same task in world frame is much
    harder.
23. **Train blind, render later.** Physics without rendering is ~10x faster;
    the video is a replay of chosen policies, not the training itself.
24. **Find joints by name at runtime** (`robot.dof_names`), never by index —
    asset updates reorder DOFs.
25. **Evolution vs gradients**: CEM (sample → select elites → refit) is real
    RL and fits in 40 lines. PPO/SAC buy sample-efficiency at the cost of
    machinery — understand this loop first and the rest is bookkeeping.

## Process

26. **Detach long jobs** (`nohup ... ; echo EXIT_$?`) and poll the exit marker;
    SSH sessions die, jobs shouldn't.
27. **Verify assets from your laptop with `curl`** against the S3 bucket —
    free, instant, no GPU billing.
28. **Change one variable per iteration** and add telemetry prints so each run
    converts one unknown into a fact. 13 iterations sounds like a lot; each
    one was cheap and conclusive.
