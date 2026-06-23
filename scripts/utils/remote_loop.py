"""Generic remote-driven evaluation loop.

One loop for every externally-driven session (autonomous eval rollouts, action
replay, DAgger collection). It is used instead of the upstream per-garment
``run_evaluation_loop`` whenever the policy declares ``uses_remote_loop`` (the
``remote`` policy). An external HTTP server owns ALL session logic — which
garment to run, how each action is produced (policy inference / pre-recorded
replay / human teleop), trajectory recording, and stop/snapshot decisions.
This loop is a thin executor that translates the server's commands into the
Isaac Sim / USD operations only the simulation process can perform.

Per episode it speaks this HTTP protocol to the server (``policy.post``):

    POST /next_task          -> {garment_name, garment_type, seed, ep_idx,
                                 restore_data?, restore_mode?, augmentation?,
                                 stop_on_success?, want_snapshot?}  | {shutdown}
    POST /reset              -> {}      (after the env has been reset + set up;
                                 body carries garment_info / augmentation_info /
                                 restore_snapshot for the server to record)
    POST /infer  {obs...}    -> {actions: [12], stop?, save_state?, n_steps?}
    POST /snapshot {state}   -> {}      (on save_state)
    POST /update_check_status-> {}      (final check status, on success)

The session ends when /next_task returns ``{"shutdown": true}``.

Everything that needs direct Isaac Sim / USD access (garment switch, physics
state restore, snapshot capture, augmentation) lives here; the server decides
*what* to do.
"""

import math

import numpy as np
import torch

from .common import stabilize_garment_after_reset as _stabilize


# Per-arm rest pose (SO101_FOLLOWER_REST_POSE_RANGE centers), degrees -> radians.
# Single source of truth for both state restore and arm parking.
_REST_POSE_DEG = [0.0, -100.0, 90.0, 50.0, 0.0, -10.0]
_REST_POSE_RAD = [d * math.pi / 180.0 for d in _REST_POSE_DEG]


def _np_to_list(obj):
    """Recursively convert numpy arrays to lists for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _np_to_list(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_np_to_list(x) for x in obj]
    return obj


def capture_physics_state(env) -> dict:
    """Snapshot garment particles/velocities/pose + both arms' joint positions.

    Used for failure-state hard-mining and success/semi-success replay. Returns
    a JSON-safe dict (lists, not arrays).
    """
    garment = env.object
    state: dict = {}
    points_attr = garment._prim.GetAttribute("points")
    if points_attr and points_attr.HasValue():
        state["garment_points"] = np.array(points_attr.Get(), dtype=np.float32).tolist()
    vel_attr = garment._prim.GetAttribute("velocities")
    if vel_attr and vel_attr.HasValue():
        state["garment_velocities"] = np.array(vel_attr.Get(), dtype=np.float32).tolist()
    pos, ori = garment.get_world_pose()
    state["garment_pos"] = np.array(pos).flatten().tolist()
    state["garment_ori"] = np.array(ori).flatten().tolist()
    state["left_joint_pos"] = env.left_arm.data.joint_pos[0].detach().cpu().numpy().tolist()
    state["right_joint_pos"] = env.right_arm.data.joint_pos[0].detach().cpu().numpy().tolist()
    return state


def restore_physics_state(env, data: dict, mode: str, label: str = "remote") -> bool:
    """Restore garment particles + arm poses from a snapshot.

    mode="lifted"  — hard-mining: lift particles +20 cm (so they don't stick to
                     the table), zero velocities, park both arms at the rest
                     pose, then settle 30 steps. Robust, slightly divergent.
    mode="exact"   — success/semi-success replay: restore particles, velocities
                     and joint positions exactly, no Z lift, no settle, so the
                     replayed action stream reproduces the original trajectory.

    Returns False (and skips) if the snapshot's particle count does not match
    the currently loaded garment; True otherwise.
    """
    from pxr import Vt

    garment = env.object
    device = env.left_arm.data.joint_pos.device
    lifted = (mode == "lifted")

    gpos = data.get("garment_pos")
    gori = data.get("garment_ori")
    if gpos is not None and gori is not None:
        garment.set_world_pose(
            position=np.array(gpos, dtype=np.float32).flatten(),
            orientation=np.array(gori, dtype=np.float32).flatten(),
        )

    pts = data.get("garment_points")
    if pts is not None:
        pts_np = np.array(pts, dtype=np.float32)
        current = np.array(garment._prim.GetAttribute("points").Get())
        if len(pts_np) != len(current):
            print(f"[{label}] restore: particle count mismatch "
                  f"(snapshot={len(pts_np)}, garment={len(current)}) — skipping",
                  flush=True)
            return False
        if lifted:
            pts_np[:, 2] += 0.20
        garment._prim.GetAttribute("points").Set(Vt.Vec3fArray.FromNumpy(pts_np))

    vel_attr = garment._prim.GetAttribute("velocities")
    if vel_attr and vel_attr.HasValue():
        saved_vel = data.get("garment_velocities")
        if not lifted and saved_vel is not None:
            vel_np = np.array(saved_vel, dtype=np.float32)
        else:
            vel_np = np.zeros((len(np.array(vel_attr.Get())), 3), dtype=np.float32)
        vel_attr.Set(Vt.Vec3fArray.FromNumpy(vel_np))

    def _set_arm(arm, joint_list):
        t = torch.tensor(np.array(joint_list, dtype=np.float32).reshape(1, -1),
                         dtype=torch.float32, device=device)
        arm.set_joint_position_target(t)
        arm.write_joint_state_to_sim(t, t * 0)

    if lifted:
        _set_arm(env.left_arm, [_REST_POSE_RAD])
        _set_arm(env.right_arm, [_REST_POSE_RAD])
        zero = torch.zeros(1, 12, dtype=torch.float32, device=device)
        for _ in range(30):
            env.step(zero)
    else:
        if data.get("left_joint_pos") is not None:
            _set_arm(env.left_arm, data["left_joint_pos"])
        if data.get("right_joint_pos") is not None:
            _set_arm(env.right_arm, data["right_joint_pos"])

    env.sim.render()
    print(f"[{label}] restored physics state "
          f"({mode}, {len(pts) if pts is not None else 0} particles)", flush=True)
    return True


def run_remote_loop(env, policy, args, augmentor, label="remote"):
    """Drive the simulation from remote commands until the server shuts down."""
    policy._env = env
    aug_enabled = getattr(augmentor, "enabled", False)
    current_garment = env.cfg.garment_name

    # init_scale drifts if scale augmentation / restore multiply it every
    # episode; snapshot the clean value so we can restore it each reset.
    if env.object is not None:
        env.object._clean_init_scale = np.array(env.object.init_scale, copy=True)

    episode = 0
    while True:
        task = policy.post("/next_task", {})
        if task.get("shutdown") or task.get("command") == "shutdown":
            print(f"[{label}] server requested shutdown", flush=True)
            break

        # --- garment switch (in place, ~5 s vs a ~2 min sim restart) -------
        new_garment = task.get("garment_name")
        if new_garment and new_garment != current_garment:
            print(f"[{label}] switch garment {current_garment} -> {new_garment}", flush=True)
            env.switch_garment(new_garment)
            current_garment = new_garment
            if env.object is not None:
                env.object._clean_init_scale = np.array(env.object.init_scale, copy=True)
            augmentor.orig_roughness.clear()  # old materials are gone
        if new_garment:
            policy._garment_name = new_garment

        # --- seed + reset --------------------------------------------------
        seed = task.get("seed")
        if seed is not None:
            try:
                env.cfg.seed = int(seed)
                env.cfg.random_seed = int(seed)
            except Exception:
                pass
        elif getattr(args, "same_seed", False) and args.seed is not None:
            env.cfg.seed = int(args.seed)
            env.cfg.random_seed = int(args.seed)

        env.reset()
        try:
            garment_info = _np_to_list({
                "object_initial_pose": env.get_all_pose(),
                "garment_name": getattr(env.cfg, "garment_name", None),
            })
        except Exception:
            garment_info = None
        policy.reset()
        _stabilize(env, args)

        # env.reset() does not reset init_scale (only switch_garment recreates
        # the object) — restore the clean value so success thresholds don't drift.
        if env.object is not None and hasattr(env.object, "_clean_init_scale"):
            env.object.init_scale = np.array(env.object._clean_init_scale, copy=True)

        # --- restore physics state (hard-mining / replay) ------------------
        restore_data = task.get("restore_data")
        # exact restore for replays (server marks them); lifted otherwise.
        is_replay = bool(
            restore_data
            and ("_replay_actions" in restore_data
                 or restore_data.get("_semi_success_replay"))
        )
        restore_mode = task.get("restore_mode", "exact" if is_replay else "lifted")
        restore_ok = None
        if restore_data:
            restore_ok = restore_physics_state(env, restore_data, restore_mode, label)
            rsf = restore_data.get("scale_factor") or restore_data.get("_saved_scale_factor")
            if rsf is not None and env.object is not None:
                env.object.init_scale = np.array(env.object.init_scale) * rsf

        # --- augmentation --------------------------------------------------
        aug_info = None
        if aug_enabled:
            # dict = restore exact saved visuals; None = randomize fresh.
            saved_aug = task.get("augmentation") or (restore_data or {}).get("augmentation")
            aug_info = augmentor.apply_episode(env, saved_aug, visual_only=is_replay)
            env.step(torch.zeros(1, 12, device=args.device))

        # Extra physics settle (e.g. after a rotation-offset z-lift). Skipped
        # for exact replays so the restored particle state isn't perturbed.
        if restore_mode != "exact":
            for _ in range(20):
                env.sim.step(render=False)

        env.clear_check_status_cache()
        if augmentor.collect_color_inputs:
            augmentor.collect_color_inputs(env)

        # --- tell the server the episode is set up -------------------------
        reset_body = {"ep_idx": task.get("ep_idx")}
        if garment_info is not None:
            reset_body["garment_info"] = garment_info
        if aug_info:
            reset_body["augmentation_info"] = aug_info
        if restore_ok is not None:
            reset_body["restore_ok"] = restore_ok
        if task.get("want_snapshot"):
            reset_body["restore_snapshot"] = capture_physics_state(env)
        policy.post("/reset", reset_body)

        # --- step loop -----------------------------------------------------
        metrics = _run_steps(
            env, policy, args, augmentor, label,
            stop_on_success=bool(task.get("stop_on_success", True)),
        )
        episode += 1
        print(f"Episode {episode} ({current_garment}): "
              f"Return={metrics['return']:.2f}, Length={metrics['length']}, "
              f"Success={metrics['success']}", flush=True)

    return []


def _run_steps(env, policy, args, augmentor, label, *, stop_on_success):
    """Inner per-step loop: post observation, receive + apply one action."""
    obs = env._get_observations()
    episode_return = 0.0
    length = 0
    success = False

    # max_steps is a hard safety bound; the server normally stops sooner.
    while length < args.max_steps:
        # Per-step stdout heartbeat: the worker that launched the sim watches
        # stdout for liveness and kills a silent process, so an episode that
        # prints nothing would be reaped mid-rollout. Dense early so a stall is
        # pinpointed, then sparse.
        if length < 30 or length % 50 == 0:
            print(f"[{label}] step {length}", flush=True)

        resp = policy.post("/infer", policy.serialize_observation(obs))

        if resp.get("save_state"):
            policy.post("/snapshot", {"state": capture_physics_state(env)})
        if resp.get("stop"):
            break

        action_np = np.array(resp["actions"], dtype=np.float32)
        action = torch.from_numpy(action_np).float().to(args.device).unsqueeze(0)

        # Per-step visual augmentation re-randomizes on wrist-render ticks
        # (length % 3 == 1): the post-reset env.step(zeros) sets the wrist timer
        # to 1/90, the 20 non-rendering settle steps don't advance it, so the
        # wrist next updates when length % 3 == 2 — randomizing on length % 3 == 1
        # keeps top and wrist seeing the same augmentation.
        if length % 3 == 1:
            for hook in (augmentor.randomize_step_color,
                         augmentor.randomize_step_light):
                if hook:
                    hook()

        for _ in range(int(resp.get("n_steps", 1))):
            env.step(action)
        reward = env.reward_buf
        episode_return += reward.item() if isinstance(reward, torch.Tensor) else float(reward)
        length += 1
        obs = env.obs_buf

        if stop_on_success and env._get_success().item():
            success = True
            # Record the final frame's check status so dense reward credits the
            # last executed action (the server records it on the last frame).
            cs = obs.get("check_status") if isinstance(obs, dict) else None
            cd = obs.get("check_distances") if isinstance(obs, dict) else None
            body = {}
            if cs is not None:
                body["check_status"] = cs.tolist() if hasattr(cs, "tolist") else list(cs)
            if cd is not None:
                body["check_distances"] = cd.tolist() if hasattr(cd, "tolist") else list(cd)
            policy.post("/update_check_status", body)
            break

    return {"return": episode_return, "length": length, "success": success}
