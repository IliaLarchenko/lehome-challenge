"""Visual augmentation engine for proxy-driven eval rollouts.

Episode-level augmentations (garment pattern swap / LAB color remap, pose /
rotation / scale offsets, roughness, camera pos+rot+focal jitter, constant
top-camera offset, arm-base shift, table-cover UV transform, dome-light
rotation, arm recolor) plus per-step randomizers (garment color tint, light
intensity / color / temperature).

Enabled when LEHOME_GARMENT_AUGMENTATION=1; all knobs come from the
LEHOME_AUG_CONFIG JSON env var (zero / absent values mean "off"). Saved
augmentation dicts round-trip through failure-state NPZs so hard-mining
replays can reproduce the exact same visuals (``saved_aug``); state replays
pass ``visual_only=True`` to skip physics-affecting offsets.

Use :func:`build_visual_augmentation` to get a hook namespace; inactive hooks
are ``None`` so callers can use simple truthiness checks.
"""

import os
from types import SimpleNamespace


def build_visual_augmentation(label: str = "eval") -> SimpleNamespace:
    """Build the augmentation hooks for one eval session.

    Returns a namespace with:
        enabled                  bool
        apply_episode            fn(env, saved_aug=None, visual_only=False) -> dict | None
        collect_color_inputs     fn(env) | None      (per-step tint: cache shader inputs)
        randomize_step_color     fn() | None
        randomize_step_light     fn() | None
        orig_roughness           dict (cleared by callers on garment switch)
    """
    # Per-step hook state (populated below when augmentation is enabled)
    _color_inputs = []
    _randomize_step_color = None
    _collect_color_inputs = None
    _randomize_step_light = None
    _apply_garment_augmentation = None

    # Bound unconditionally so callers can clear it even when augmentation is
    # disabled (first garment switch would otherwise crash on a missing dict).
    _orig_roughness = {}

    enabled = os.environ.get("LEHOME_GARMENT_AUGMENTATION") == "1"
    if enabled:
        import json as _json_aug_cfg
        import random as _rng_aug
        import numpy as _np_aug
        try:
            import cv2 as _cv2_aug
            _aug_has_cv2 = True
        except ImportError:
            _aug_has_cv2 = False
        from pxr import Usd as _Usd_aug, UsdShade as _UsdShade_aug, Sdf as _Sdf_aug
        import omni.kit.commands as _omni_cmd_aug
        import isaacsim.core.utils.prims as _prims_aug
        from isaacsim.core.utils.stage import add_reference_to_stage as _add_ref_aug

        # Parse augmentation config
        _aug_cfg_raw = os.environ.get("LEHOME_AUG_CONFIG", "{}")
        _aug_cfg = _json_aug_cfg.loads(_aug_cfg_raw)

        # Episode-level params
        _aug_p_pattern = float(_aug_cfg.get("pattern_swap_p", 0.0))
        _aug_p_color = float(_aug_cfg.get("color_remap_p", 0.0))
        _aug_pos_offset = float(_aug_cfg.get("pos_offset_range", 0.0))
        _aug_rot_offset = float(_aug_cfg.get("rot_offset_range", 0.0))
        _aug_scale_range = float(_aug_cfg.get("scale_range", 0.0))
        _aug_roughness_range = float(_aug_cfg.get("roughness_range", 0.0))
        _aug_cam_pos_jitter = float(_aug_cfg.get("camera_pos_jitter", 0.0))
        _aug_cam_rot_jitter = float(_aug_cfg.get("camera_rot_jitter", 0.0))
        _aug_top_cam_pos_jitter = float(_aug_cfg.get("top_camera_pos_jitter", 0.0))
        _aug_top_cam_rot_jitter = float(_aug_cfg.get("top_camera_rot_jitter", 0.0))
        # Constant (non-random) offset applied on top of any jitter.
        _aug_top_cam_pos_offset = tuple(_aug_cfg.get("top_camera_pos_offset", (0.0, 0.0, 0.0)))
        _aug_top_cam_rot_offset_deg = tuple(_aug_cfg.get("top_camera_rot_offset_deg", (0.0, 0.0, 0.0)))
        # Constant focal-length multiplier on the top camera (1.0 = no
        # change). Applied once per episode on top of the saved base focal
        # length; stacks with camera_focal_jitter if both are active.
        _aug_top_cam_focal_scale = float(_aug_cfg.get("top_camera_focal_scale", 1.0))
        _aug_arm_xy_shift = float(_aug_cfg.get("arm_xy_shift", 0.0))
        _aug_arm_rot_z_deg = float(_aug_cfg.get("arm_rot_z_deg", 0.0))
        _aug_table_uv_shift = float(_aug_cfg.get("table_uv_shift", 0.0))
        _aug_table_uv_rot_deg = float(_aug_cfg.get("table_uv_rot_deg", 0.0))
        _aug_cam_focal_jitter = float(_aug_cfg.get("camera_focal_jitter", 0.0))
        _aug_dome_rot_deg = float(_aug_cfg.get("light_dome_rot_deg", 0.0))
        # Per-step params
        _aug_step_color_tint = bool(_aug_cfg.get("step_color_tint", True))
        _aug_light_intensity = float(_aug_cfg.get("light_intensity_range", 0.0))
        _aug_light_color = float(_aug_cfg.get("light_color_range", 0.0))
        _aug_arm_color_range = float(_aug_cfg.get("arm_color_range", 0.0))
        _aug_light_color_temp = float(_aug_cfg.get("light_color_temp_range", 0.0))
        # (no table tint knob — unreliable across USD material types)

        # Env-var overrides take precedence over the JSON config when set.
        if os.environ.get("LEHOME_PATTERN_AUG_P"):
            _aug_p_pattern = float(os.environ["LEHOME_PATTERN_AUG_P"])
        if os.environ.get("LEHOME_COLOR_AUG_P"):
            _aug_p_color = float(os.environ["LEHOME_COLOR_AUG_P"])

        # --- Texture pool for pattern swap ---
        _tex_dir_aug = os.path.join(
            os.getcwd(), "Assets", "objects", "Challenge_Garment", "Release", "Color_Texture"
        )
        _all_tex_usds = sorted([
            os.path.join(_tex_dir_aug, f)
            for f in os.listdir(_tex_dir_aug) if f.endswith(".usd")
        ])
        _tex_pool = {}
        _tex_pool_ready = [False]

        def _init_tex_pool(stage):
            if _tex_pool_ready[0]:
                return
            for _i, _usd in enumerate(_all_tex_usds):
                _pp = f"/World/_aug_tex_pool/t{_i}"
                _add_ref_aug(usd_path=_usd, prim_path=_pp)
                _vis = _prims_aug.get_prim_at_path(_pp)
                _ch = _prims_aug.get_prim_children(_vis)
                if _ch:
                    _tex_pool[_i] = (_ch[0].GetPath(), _usd)
            _tex_pool_ready[0] = True

        # --- LAB color remap ---
        _aug_temp_dir = [None]
        _aug_known_temps = set()

        def _get_aug_temp_dir():
            if _aug_temp_dir[0] is None:
                import tempfile as _tf
                _aug_temp_dir[0] = _tf.mkdtemp(prefix="garment_aug_")
            return _aug_temp_dir[0]

        def _find_base_color(tex_usd):
            if tex_usd is None:
                return None
            _p = tex_usd
            if _p.startswith("/"):
                _p = os.path.join(os.getcwd(), _p[1:])
            _base = os.path.splitext(_p)[0]
            _dir = _base.replace("_JPG", "-JPG")
            _bc = os.path.join(_dir, "BaseColor.jpg")
            return _bc if os.path.exists(_bc) else None

        def _remap_texture(bc_path, seed):
            _img = _cv2_aug.imread(bc_path)
            if _img is None:
                return None
            _lab = _cv2_aug.cvtColor(_img, _cv2_aug.COLOR_BGR2LAB).astype(_np_aug.float32)
            _h, _w = _lab.shape[:2]
            _ab = _lab[:, :, 1:3].reshape(-1, 2)
            _n_levels = 16
            _bin_idx = _np_aug.clip((_ab * _n_levels / 256.0).astype(_np_aug.int32), 0, _n_levels - 1)
            _bin_keys = _bin_idx[:, 0] * _n_levels + _bin_idx[:, 1]
            _centers = {}
            for _bk in _np_aug.unique(_bin_keys):
                _mask = _bin_keys == _bk
                _centers[int(_bk)] = _ab[_mask].mean(axis=0)
            _rng = _np_aug.random.RandomState(seed)
            _shifts = {}
            for _bk, _center in sorted(_centers.items()):
                _target = _rng.uniform(20, 235, size=2).astype(_np_aug.float32)
                _shifts[_bk] = _target - _center
            _res_ab = _ab.copy()
            for _bk, _shift in _shifts.items():
                _mask = _bin_keys == _bk
                _res_ab[_mask] += _shift
            _res_lab = _lab.reshape(-1, 3).copy()
            _res_lab[:, 1:3] = _res_ab.clip(0, 255)
            _res_lab = _res_lab.reshape(_h, _w, 3).astype(_np_aug.uint8)
            _res_bgr = _cv2_aug.cvtColor(_res_lab, _cv2_aug.COLOR_LAB2BGR)
            import hashlib as _hl
            _nh = _hl.md5(bc_path.encode()).hexdigest()[:8]
            _out = os.path.join(_get_aug_temp_dir(), f"{_nh}_s{seed}.jpg")
            _cv2_aug.imwrite(_out, _res_bgr, [_cv2_aug.IMWRITE_JPEG_QUALITY, 95])
            _aug_known_temps.add(_out)
            return _out

        def _get_submesh_targets(obj):
            """Return [(submesh_prim_path, texture_slot_index), ...]."""
            _orig = list(getattr(obj, "visual_usd_paths", []) or [])
            _mp = _prims_aug.get_prim_at_path(obj.mesh_prim_path)
            _subs = _prims_aug.get_prim_children(_mp)
            if len(_subs) == 0:
                return [(obj.mesh_prim_path, 0)], _orig
            _targets = []
            for _sub in _subs:
                _nm = _sub.GetName()
                _idx = 0
                if _nm == "mesh":
                    _idx = 0
                elif _nm.startswith("mesh"):
                    try:
                        _idx = int(_nm.replace("mesh", ""))
                    except ValueError:
                        _idx = 0
                _targets.append((_sub.GetPath(), _idx))
            return _targets, _orig

        def _update_base_color_shader(stage, sub_path, new_path):
            """Find the BaseColor texture reader on a submesh's material and update it."""
            _sub_prim = stage.GetPrimAtPath(sub_path)
            _binding = _UsdShade_aug.MaterialBindingAPI(_sub_prim)
            _mat, _ = _binding.ComputeBoundMaterial()
            if not _mat:
                return
            for _desc in _Usd_aug.PrimRange(_mat.GetPrim()):
                if _desc.GetTypeName() != "Shader":
                    continue
                _sh = _UsdShade_aug.Shader(_desc)
                _fi = _sh.GetInput("file")
                if not _fi:
                    continue
                _fv = _fi.Get()
                if _fv is None:
                    continue
                _fs = str(_fv.resolvedPath or _fv.path)
                if "Color" in _fs or "color" in _fs or _fs in _aug_known_temps:
                    _fi.Set(_Sdf_aug.AssetPath(new_path))
                    return

        # --- Quaternion multiply helper ---
        def _qmul(q1, q2):
            """Hamilton product of two quaternions [w, x, y, z]."""
            w1, x1, y1, z1 = q1
            w2, x2, y2, z2 = q2
            return [
                w1*w2 - x1*x2 - y1*y2 - z1*z2,
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2,
                w1*z2 + x1*y2 - y1*x2 + z1*w2,
            ]

        # _orig_roughness is initialized unconditionally above the
        # augmentation-setup block, so callers can clear it even when
        # augmentation is disabled.

        # --- Camera base poses (saved once for jitter restore) ---
        _camera_base_poses = {}  # camera_name -> (pos, rot)
        _camera_base_focal = {}  # camera_name -> base focalLength (mm)

        def _save_camera_base_poses(env):
            """Save original camera USD prim xform + focal length (for restore).

            Cameras are spawned with offset baked into the prim's xformOps.
            We jitter by modifying the prim's translate/orient/focalLength directly.
            """
            if _camera_base_poses:
                return
            from pxr import UsdGeom as _UsdGeom_cam
            _stage = env.scene.stage
            for _cam_name in ("left_camera", "right_camera", "top_camera"):
                _cam = getattr(env, _cam_name, None)
                if _cam is None:
                    continue
                try:
                    _cam_prim = _stage.GetPrimAtPath(_cam.cfg.prim_path)
                    if not _cam_prim.IsValid():
                        print(f"[{label}] P8: Camera {_cam_name} prim not found at {_cam.cfg.prim_path}", flush=True)
                        continue
                    _xf = _UsdGeom_cam.Xformable(_cam_prim)
                    _xf_ops = _xf.GetOrderedXformOps()
                    _translate_op = None
                    _orient_op = None
                    for _op in _xf_ops:
                        if _op.GetOpType() == _UsdGeom_cam.XformOp.TypeTranslate:
                            _translate_op = _op
                        elif _op.GetOpType() == _UsdGeom_cam.XformOp.TypeOrient:
                            _orient_op = _op
                    if _translate_op:
                        _base_t = _translate_op.Get()
                        _camera_base_poses[_cam_name] = (
                            _translate_op,
                            tuple(_base_t) if _base_t else (0, 0, 0),
                            _orient_op,
                            tuple([_orient_op.Get().GetReal()] + list(_orient_op.Get().GetImaginary())) if _orient_op and _orient_op.Get() else (1, 0, 0, 0),
                        )
                        print(f"[{label}] P8: Camera {_cam_name} xform: pos={_base_t}", flush=True)
                    else:
                        print(f"[{label}] P8: Camera {_cam_name} has no translate xformOp", flush=True)
                    _fa = _cam_prim.GetAttribute("focalLength")
                    if _fa and _fa.IsValid():
                        _fv = _fa.Get()
                        if _fv is not None:
                            _camera_base_focal[_cam_name] = float(_fv)
                except Exception as _e_cam_save:
                    print(f"[{label}] P8: Failed to save camera {_cam_name}: {_e_cam_save}", flush=True)

        def _apply_garment_augmentation(env, saved_aug=None, visual_only=False):
            """Apply all episode-level augmentations.

            saved_aug: if provided, restore exact augmentation from failure state.
            visual_only: if True, skip physics-affecting augmentations (pos, rot, scale).
                Used for success/semi-success replay where exact particle state is restored.
            Returns dict with augmentation info or None if nothing changed.
            """
            _obj = env.object
            if _obj is None:
                return None
            _stage = env.scene.stage
            _orig_paths = list(getattr(_obj, "visual_usd_paths", []) or [])

            # Restore exact saved augmentation
            if saved_aug:
                _do_pattern = saved_aug.get("pattern_swap", False)
                _do_color = saved_aug.get("color_remap", False)
                _pat_indices = saved_aug.get("pattern_indices")
                _color_seed = saved_aug.get("color_seed")
                _saved_scale = saved_aug.get("scale_factor")
                _saved_pos_off = saved_aug.get("pos_offset")
                _saved_rot_off = saved_aug.get("rot_offset")
                _saved_roughness_delta = saved_aug.get("roughness_delta")
                _saved_cam_jitter = saved_aug.get("camera_jitter")
                _saved_arm_shift = saved_aug.get("arm_shift")
                _saved_table_uv = saved_aug.get("table_uv")
                _saved_cam_focal = saved_aug.get("camera_focal")
                _saved_dome_rot = saved_aug.get("dome_rot")
            else:
                # Roll dice for each augmentation independently
                _do_pattern = _rng_aug.random() < _aug_p_pattern if _aug_p_pattern > 0 else False
                _do_color = _aug_has_cv2 and _rng_aug.random() < _aug_p_color if _aug_p_color > 0 else False
                _pat_indices = None
                _color_seed = None
                _saved_scale = None
                _saved_pos_off = None
                _saved_rot_off = None
                _saved_roughness_delta = None
                _saved_cam_jitter = None
                _saved_arm_shift = None
                _saved_table_uv = None
                _saved_cam_focal = None
                _saved_dome_rot = None

            _aug_info = {}

            # --- Step 1: Pattern swap (rebind to random textures from pool) ---
            _targets, _orig = _get_submesh_targets(_obj) if _orig_paths else ([], [])
            _unique_orig = list(dict.fromkeys(_orig))
            _active_tex = dict(zip(_unique_orig, _unique_orig))
            if _do_pattern and _orig_paths:
                _init_tex_pool(_stage)
                _pool_keys = list(_tex_pool.keys())
                if _pool_keys:
                    if _pat_indices is None:
                        _pat_indices = [_rng_aug.choice(_pool_keys) for _ in range(len(_unique_orig))]
                    for _i, _otex in enumerate(_unique_orig):
                        _pi = _pat_indices[_i % len(_pat_indices)]
                        if _pi in _tex_pool:
                            _mat_path, _new_usd = _tex_pool[_pi]
                            _active_tex[_otex] = _new_usd
                            for _sub_path, _tidx in _targets:
                                if _tidx < len(_orig) and _orig[_tidx] == _otex:
                                    _omni_cmd_aug.execute(
                                        "BindMaterialCommand",
                                        prim_path=_sub_path,
                                        material_path=_mat_path,
                                    )
                _aug_info["pattern_swap"] = True
                _aug_info["pattern_indices"] = _pat_indices

            # --- Step 2: Color remap (LAB space) ---
            if _do_color and _aug_has_cv2 and _orig_paths:
                if _color_seed is None:
                    _color_seed = _rng_aug.randint(0, 2**31 - 1)
                _unique_active = list(dict.fromkeys(_active_tex.values()))
                _remapped = {}
                for _i, _tex in enumerate(_unique_active):
                    _bc = _find_base_color(_tex)
                    if _bc:
                        _tmp = _remap_texture(_bc, _color_seed + _i)
                        if _tmp:
                            _remapped[_tex] = _tmp
                if _remapped:
                    _done_mats = set()
                    for _sub_path, _tidx in _targets:
                        _otex = _orig[_tidx] if _tidx < len(_orig) else None
                        _atex = _active_tex.get(_otex, _otex)
                        if _atex not in _remapped:
                            continue
                        _mk = str(_sub_path)
                        if _mk in _done_mats:
                            continue
                        _update_base_color_shader(_stage, _sub_path, _remapped[_atex])
                        _done_mats.add(_mk)
                _aug_info["color_remap"] = True
                _aug_info["color_seed"] = _color_seed

            # --- Steps 3-5: Physics-affecting augmentations (skip for state replay) ---
            # --- Step 3: Garment position offset ---
            # Z offset is always non-negative (garment starts above table, only raise it)
            if _aug_pos_offset > 0 and not visual_only:
                if _saved_pos_off is not None:
                    _pos_off = _saved_pos_off
                else:
                    _pos_off = [
                        _rng_aug.uniform(-_aug_pos_offset, _aug_pos_offset),
                        _rng_aug.uniform(-_aug_pos_offset, _aug_pos_offset),
                        _rng_aug.uniform(0, _aug_pos_offset),  # z: only up, never reduce
                    ]
                try:
                    _cur_pos, _cur_ori = _obj.get_world_pose()
                    _new_pos = [_cur_pos[i] + _pos_off[i] for i in range(3)]
                    _obj.set_world_pose(_new_pos, _cur_ori)
                    _aug_info["pos_offset"] = _pos_off
                except Exception as _e_pos:
                    print(f"[{label}] Aug pos_offset error: {_e_pos}", flush=True)

            # --- Step 4: Garment rotation offset ---
            if _aug_rot_offset > 0 and not visual_only:
                if _saved_rot_off is not None:
                    _rot_off = _saved_rot_off
                else:
                    _rot_off = [_rng_aug.uniform(-_aug_rot_offset, _aug_rot_offset) for _ in range(3)]
                try:
                    from isaacsim.core.utils.rotations import euler_angles_to_quat as _eaq
                    _cur_pos, _cur_ori = _obj.get_world_pose()
                    _delta_q = _eaq(_rot_off, degrees=True)
                    _new_ori = _qmul(list(_delta_q), list(_cur_ori))
                    _obj.set_world_pose(list(_cur_pos), _new_ori)
                    _aug_info["rot_offset"] = _rot_off
                    # Z-lift proportional to tilt: max(|rx|,|ry|) * 2 cm
                    _tilt_deg = max(abs(_rot_off[0]), abs(_rot_off[1]))
                    _z_lift = _tilt_deg * 0.02  # 2cm per degree
                    if _z_lift > 0.001:
                        _cur_pos2, _cur_ori2 = _obj.get_world_pose()
                        _obj.set_world_pose([_cur_pos2[0], _cur_pos2[1], _cur_pos2[2] + _z_lift], _cur_ori2)
                        print(f"[{label}] Aug rot tilt={_tilt_deg:.1f}° — z-lift +{_z_lift*100:.1f}cm", flush=True)
                except Exception as _e_rot:
                    print(f"[{label}] Aug rot_offset error: {_e_rot}", flush=True)

            # --- Step 5: Garment scale (with reward threshold auto-adjustment) ---
            if _aug_scale_range > 0 and not visual_only:
                if _saved_scale is not None:
                    _scale_f = _saved_scale
                else:
                    _scale_f = 1.0 + _rng_aug.uniform(-_aug_scale_range, _aug_scale_range)
                try:
                    # Scale particle positions around centroid
                    from pxr import Vt as _Vt_scale
                    _pts_attr = _obj._prim.GetAttribute("points")
                    _pts = _np_aug.array(_pts_attr.Get())
                    _centroid = _pts.mean(axis=0)
                    _pts_scaled = _centroid + (_pts - _centroid) * _scale_f
                    _pts_attr.Set(_Vt_scale.Vec3fArray.FromNumpy(_pts_scaled.astype(_np_aug.float32)))
                    # Update init_scale so success thresholds auto-adjust to the new scale
                    _obj.init_scale = _np_aug.array(_obj.init_scale) * _scale_f
                    _aug_info["scale_factor"] = _scale_f
                    print(f"[{label}] Aug scale: {_scale_f:.3f} (init_scale now {_obj.init_scale})", flush=True)
                except Exception as _e_sc:
                    print(f"[{label}] Aug scale error: {_e_sc}", flush=True)

            # --- Step 6: Garment roughness variation ---
            # Use saved original roughness values to avoid accumulation
            # across episodes (the shader input persists across reset()).
            if _aug_roughness_range > 0 and _orig_paths:
                if _saved_roughness_delta is not None:
                    _rough_delta = _saved_roughness_delta
                else:
                    _rough_delta = _rng_aug.uniform(-_aug_roughness_range, _aug_roughness_range)
                try:
                    _seen_rough_mats = set()
                    for _sub_path, _ in _targets:
                        _sub_prim = _stage.GetPrimAtPath(_sub_path)
                        _binding = _UsdShade_aug.MaterialBindingAPI(_sub_prim)
                        _mat, _ = _binding.ComputeBoundMaterial()
                        if not _mat:
                            continue
                        _mp = str(_mat.GetPath())
                        if _mp in _seen_rough_mats:
                            continue
                        _seen_rough_mats.add(_mp)
                        for _desc in _Usd_aug.PrimRange(_mat.GetPrim()):
                            if _desc.GetTypeName() != "Shader":
                                continue
                            _sh = _UsdShade_aug.Shader(_desc)
                            _ri = _sh.GetInput("roughness")
                            if _ri:
                                _rv = _ri.Get()
                                if _rv is not None:
                                    _rkey = str(_desc.GetPath())
                                    # Save original roughness on first encounter
                                    if _rkey not in _orig_roughness:
                                        _orig_roughness[_rkey] = float(_rv)
                                    _base_rv = _orig_roughness[_rkey]
                                    _ri.Set(max(0.0, min(1.0, _base_rv + _rough_delta)))
                    _aug_info["roughness_delta"] = _rough_delta
                except Exception as _e_rough:
                    print(f"[{label}] Aug roughness error: {_e_rough}", flush=True)

            # --- Step 7: Camera position/rotation jitter ---
            _aug_top_cam_has_const = (
                any(abs(v) > 0 for v in _aug_top_cam_pos_offset)
                or any(abs(v) > 0 for v in _aug_top_cam_rot_offset_deg)
            )
            _any_cam_jitter = (
                _aug_cam_pos_jitter > 0 or _aug_cam_rot_jitter > 0
                or _aug_top_cam_pos_jitter > 0 or _aug_top_cam_rot_jitter > 0
                or _aug_top_cam_has_const
            )
            if _any_cam_jitter:
                _save_camera_base_poses(env)
                if _saved_cam_jitter is not None:
                    _cam_jit = _saved_cam_jitter
                else:
                    _cam_jit = {}
                    for _cn in _camera_base_poses:
                        _is_top = (_cn == "top_camera")
                        _pj = _aug_top_cam_pos_jitter if _is_top else _aug_cam_pos_jitter
                        _rj = _aug_top_cam_rot_jitter if _is_top else _aug_cam_rot_jitter
                        _dp = [_rng_aug.uniform(-_pj, _pj) for _ in range(3)] if _pj > 0 else [0,0,0]
                        _dr = [_rng_aug.uniform(-_rj, _rj) for _ in range(3)] if _rj > 0 else [0,0,0]
                        # Add constant (non-random) offset on top of the random jitter
                        # for the top camera only.
                        if _is_top:
                            _dp = [_dp[i] + float(_aug_top_cam_pos_offset[i]) for i in range(3)]
                            _dr = [_dr[i] + float(_aug_top_cam_rot_offset_deg[i]) for i in range(3)]
                        _cam_jit[_cn] = {"dp": _dp, "dr": _dr}
                try:
                    from pxr import Gf as _Gf_cam
                    for _cn, (_t_op, _base_pos, _o_op, _base_rot) in _camera_base_poses.items():
                        if _cn not in _cam_jit:
                            continue
                        _dj = _cam_jit[_cn]
                        # Jitter position: add delta to base translate
                        if _t_op is not None:
                            _new_p = _Gf_cam.Vec3d(
                                _base_pos[0] + _dj["dp"][0],
                                _base_pos[1] + _dj["dp"][1],
                                _base_pos[2] + _dj["dp"][2],
                            )
                            _t_op.Set(_new_p)
                        # Jitter rotation: compose delta quat with base orient
                        if _o_op is not None and any(abs(d) > 0 for d in _dj["dr"]):
                            from isaacsim.core.utils.rotations import euler_angles_to_quat as _eaq_cam
                            _dq = _eaq_cam(_dj["dr"], degrees=True)
                            _nq = _qmul(list(_dq), list(_base_rot))
                            _o_op.Set(_Gf_cam.Quatd(_nq[0], _Gf_cam.Vec3d(_nq[1], _nq[2], _nq[3])))
                    _aug_info["camera_jitter"] = _cam_jit
                except Exception as _e_cam:
                    print(f"[{label}] Aug camera jitter error: {_e_cam}", flush=True)

            # --- Step 8: Arm base shift (XY translation + Z-axis rotation) ---
            # Physics-affecting: changes the kinematic chain root, so end-effector
            # pose under the same joint targets shifts. Skipped in visual_only
            # replays. For hard mining, saved value is reused verbatim.
            _arm_shift_active = (_aug_arm_xy_shift > 0 or _aug_arm_rot_z_deg > 0) and not visual_only
            if _arm_shift_active:
                if _saved_arm_shift is not None:
                    _arm_shift = _saved_arm_shift
                else:
                    _arm_shift = {}
                    for _arm_name in ("left", "right"):
                        _arm_shift[_arm_name] = {
                            "dxy": [
                                _rng_aug.uniform(-_aug_arm_xy_shift, _aug_arm_xy_shift),
                                _rng_aug.uniform(-_aug_arm_xy_shift, _aug_arm_xy_shift),
                            ] if _aug_arm_xy_shift > 0 else [0.0, 0.0],
                            "drz": _rng_aug.uniform(-_aug_arm_rot_z_deg, _aug_arm_rot_z_deg)
                                   if _aug_arm_rot_z_deg > 0 else 0.0,
                        }
                try:
                    import math as _m_arm
                    for _arm_name, _arm_obj in (("left", getattr(env, "left_arm", None)),
                                                ("right", getattr(env, "right_arm", None))):
                        if _arm_obj is None or _arm_name not in _arm_shift:
                            continue
                        _shift = _arm_shift[_arm_name]
                        # Current root pose right after reset: [x, y, z, qw, qx, qy, qz].
                        _cur = _arm_obj.data.root_pose_w[0].clone()
                        _cur[0] = _cur[0] + float(_shift["dxy"][0])
                        _cur[1] = _cur[1] + float(_shift["dxy"][1])
                        if abs(float(_shift["drz"])) > 0:
                            _half = _m_arm.radians(float(_shift["drz"])) * 0.5
                            _dq = [_m_arm.cos(_half), 0.0, 0.0, _m_arm.sin(_half)]
                            _bq = [float(_cur[3]), float(_cur[4]), float(_cur[5]), float(_cur[6])]
                            _nq = _qmul(_dq, _bq)
                            _cur[3] = _nq[0]
                            _cur[4] = _nq[1]
                            _cur[5] = _nq[2]
                            _cur[6] = _nq[3]
                        _arm_obj.write_root_pose_to_sim(_cur.unsqueeze(0))
                    _aug_info["arm_shift"] = _arm_shift
                except Exception as _e_arm:
                    print(f"[{label}] Aug arm_shift error: {_e_arm}", flush=True)

            # --- Step 9: Table-cover UV transform (visual only, no physics) ---
            _table_uv_active = _aug_table_uv_shift > 0 or _aug_table_uv_rot_deg > 0
            if _table_uv_active:
                if _saved_table_uv is not None:
                    _tbl_uv = _saved_table_uv
                else:
                    _tbl_uv = {
                        "ds": _rng_aug.uniform(-_aug_table_uv_shift, _aug_table_uv_shift)
                              if _aug_table_uv_shift > 0 else 0.0,
                        "dt": _rng_aug.uniform(-_aug_table_uv_shift, _aug_table_uv_shift)
                              if _aug_table_uv_shift > 0 else 0.0,
                        "drot": _rng_aug.uniform(-_aug_table_uv_rot_deg, _aug_table_uv_rot_deg)
                                if _aug_table_uv_rot_deg > 0 else 0.0,
                    }
                try:
                    from pxr import Gf as _Gf_tbl
                    _tex_shader_path = "/World/Scene/scene/Table038/looks/M_Table038a/UsdPreviewSurface/________7/________7"
                    _tex_prim = _stage.GetPrimAtPath(_tex_shader_path)
                    if _tex_prim.IsValid():
                        _tex_sh = _UsdShade_aug.Shader(_tex_prim)
                        _st_inp = _tex_sh.GetInput("st")
                        _xf_path = "/World/Scene/scene/Table038/looks/M_Table038a/UsdPreviewSurface/_aug_uv_xf"
                        _xf_prim = _stage.GetPrimAtPath(_xf_path)
                        if not _xf_prim.IsValid():
                            # Insert a UsdTransform2d between the existing st source and the texture.
                            _xf_sh = _UsdShade_aug.Shader.Define(_stage, _xf_path)
                            _xf_sh.CreateIdAttr("UsdTransform2d")
                            _xf_in_inp = _xf_sh.CreateInput("in", _Sdf_aug.ValueTypeNames.Float2)
                            _xf_sh.CreateInput("translation", _Sdf_aug.ValueTypeNames.Float2).Set(_Gf_tbl.Vec2f(0.0, 0.0))
                            _xf_sh.CreateInput("rotation", _Sdf_aug.ValueTypeNames.Float).Set(0.0)
                            _xf_sh.CreateInput("scale", _Sdf_aug.ValueTypeNames.Float2).Set(_Gf_tbl.Vec2f(1.0, 1.0))
                            _xf_out = _xf_sh.CreateOutput("result", _Sdf_aug.ValueTypeNames.Float2)
                            # Rewire: forward whatever was feeding the texture's `st` into the new node's `in`.
                            # GetConnectedSources() returns (list[UsdShadeConnectionSourceInfo], list[SdfPath]).
                            # Pass the SourceInfo directly to ConnectToSource (signature 5);
                            # pass _xf_out directly back to the texture's st input (signature 1).
                            if _st_inp:
                                _srcs_tuple = _st_inp.GetConnectedSources()
                                _src_infos = _srcs_tuple[0] if _srcs_tuple else None
                                if _src_infos:
                                    _xf_in_inp.ConnectToSource(_src_infos[0])
                                _st_inp.ConnectToSource(_xf_out)
                            _xf_prim = _stage.GetPrimAtPath(_xf_path)
                        if _xf_prim.IsValid():
                            _xf_sh2 = _UsdShade_aug.Shader(_xf_prim)
                            _tr_inp = _xf_sh2.GetInput("translation")
                            _rot_inp = _xf_sh2.GetInput("rotation")
                            if _tr_inp:
                                _tr_inp.Set(_Gf_tbl.Vec2f(float(_tbl_uv["ds"]), float(_tbl_uv["dt"])))
                            if _rot_inp:
                                _rot_inp.Set(float(_tbl_uv["drot"]))
                            _aug_info["table_uv"] = _tbl_uv
                    else:
                        print(f"[{label}] Aug table_uv: shader prim not found at {_tex_shader_path}", flush=True)
                except Exception as _e_tbl:
                    print(f"[{label}] Aug table_uv error: {_e_tbl}", flush=True)

            # --- Step 10: Camera focal length jitter (visual only, per camera) ---
            # Constant top-camera focal scale (1.0 = no change) stacks
            # multiplicatively with the jitter — both default to no-op so
            # the combined block is also a no-op when neither is configured.
            _top_focal_active = abs(_aug_top_cam_focal_scale - 1.0) > 1e-6
            if _aug_cam_focal_jitter > 0 or _top_focal_active:
                _save_camera_base_poses(env)
                if _camera_base_focal:
                    if _saved_cam_focal is not None:
                        _cam_focal = _saved_cam_focal
                    else:
                        _cam_focal = {}
                        for _cn in _camera_base_focal:
                            _jit = (
                                1.0 + _rng_aug.uniform(-_aug_cam_focal_jitter, _aug_cam_focal_jitter)
                                if _aug_cam_focal_jitter > 0 else 1.0
                            )
                            _const = _aug_top_cam_focal_scale if _cn == "top_camera" else 1.0
                            _cam_focal[_cn] = float(_camera_base_focal[_cn]) * _jit * _const
                    try:
                        for _cn, _f in _cam_focal.items():
                            _cam = getattr(env, _cn, None)
                            if _cam is None:
                                continue
                            _cam_prim = _stage.GetPrimAtPath(_cam.cfg.prim_path)
                            if _cam_prim.IsValid():
                                _fa = _cam_prim.GetAttribute("focalLength")
                                if _fa:
                                    _fa.Set(float(_f))
                        _aug_info["camera_focal"] = _cam_focal
                        if _top_focal_active:
                            print(f"[{label}] Aug top_camera_focal_scale={_aug_top_cam_focal_scale:.3f}", flush=True)
                    except Exception as _e_cf:
                        print(f"[{label}] Aug camera_focal error: {_e_cf}", flush=True)

            # --- Arm color (shades of orange, shared across both arms) ---
            # Applied here, before env.step(zeros) and the settle steps: writes
            # that fire after those frames render get dropped even though the USD
            # value updates correctly, so it shares the same window as the garment
            # pattern swap / color remap. Sets diffuse_color_constant on the
            # OmniPBR.mdl shaders the SO101 arms use.
            if _aug_arm_color_range > 0:
                try:
                    # Deep fruit-orange anchor (≈1.0, 0.35, 0.02 → #FF5905)
                    # with tight per-channel jitter. Think ripe orange /
                    # persimmon — saturated, warm, no yellow drift.
                    _r_a = _rng_aug.uniform(0.95, 1.00)
                    _g_a = 0.35 + _rng_aug.uniform(-0.04, 0.04)
                    _b_a = 0.02 + _rng_aug.uniform(-0.02, 0.02)
                    _g_a = max(0.28, min(0.42, _g_a))
                    _b_a = max(0.0, min(0.06, _b_a))
                    _col_a = _Gf_tint.Vec3f(float(_r_a), float(_g_a), float(_b_a))
                    _n_arm = 0
                    _rb_a = None
                    for _ar in ("/World/Robot/Left_Robot", "/World/Robot/Right_Robot"):
                        _arm_prim_a = _stage.GetPrimAtPath(_ar)
                        if not _arm_prim_a.IsValid():
                            continue
                        for _dsh in _Usd_aug.PrimRange(_arm_prim_a):
                            if _dsh.GetTypeName() != "Shader":
                                continue
                            # Skip servo-motor housings (STS3215 black plastic);
                            # only the 3D-printed plastic body parts get recolored.
                            _sh_path = str(_dsh.GetPath())
                            if "material_sts3215" in _sh_path:
                                continue
                            try:
                                _inp = _UsdShade_aug.Shader(_dsh).CreateInput(
                                    "diffuse_color_constant", _Sdf_aug.ValueTypeNames.Color3f
                                )
                                _inp.Set(_col_a)
                                _n_arm += 1
                                if _rb_a is None:
                                    _rb_a = _inp.Get()
                            except Exception as _e_a:
                                print(f"[{label}] Aug arm_color set failed at {_dsh.GetPath()}: {_e_a}", flush=True)
                    _aug_info["arm_color"] = [float(_r_a), float(_g_a), float(_b_a)]
                    print(f"[{label}] Aug arm_color RGB=({_r_a:.2f},{_g_a:.2f},{_b_a:.2f}) "
                          f"on {_n_arm} OmniPBR shaders; readback[0]={_rb_a}", flush=True)
                except Exception as _e_ac:
                    print(f"[{label}] Aug arm_color error: {_e_ac}", flush=True)

            # --- Step 11: Dome light rotation (visual only) ---
            if _aug_dome_rot_deg > 0:
                if _saved_dome_rot is not None:
                    _dome_rot = _saved_dome_rot
                else:
                    _dome_rot = [_rng_aug.uniform(-_aug_dome_rot_deg, _aug_dome_rot_deg) for _ in range(3)]
                try:
                    from pxr import Gf as _Gf_dome, UsdGeom as _UsdGeom_dome
                    _lp = _stage.GetPrimAtPath("/World/Light")
                    if _lp.IsValid():
                        _xf = _UsdGeom_dome.Xformable(_lp)
                        _rot_op = None
                        for _op in _xf.GetOrderedXformOps():
                            if _op.GetOpType() == _UsdGeom_dome.XformOp.TypeRotateXYZ:
                                _rot_op = _op
                                break
                        if _rot_op is None:
                            _rot_op = _xf.AddRotateXYZOp()
                        _rot_op.Set(_Gf_dome.Vec3f(float(_dome_rot[0]), float(_dome_rot[1]), float(_dome_rot[2])))
                        _aug_info["dome_rot"] = _dome_rot
                except Exception as _e_dome:
                    print(f"[{label}] Aug dome_rot error: {_e_dome}", flush=True)

            return _aug_info if _aug_info else None

        print(f"[{label}] Augmentation enabled (pattern={_aug_p_pattern}, color={_aug_p_color}, "
              f"pos={_aug_pos_offset}, rot={_aug_rot_offset}, scale={_aug_scale_range}, "
              f"rough={_aug_roughness_range}, cam_pos={_aug_cam_pos_jitter}, cam_rot={_aug_cam_rot_jitter}, "
              f"top_cam_pos={_aug_top_cam_pos_jitter}, top_cam_rot={_aug_top_cam_rot_jitter}, "
              f"top_cam_focal_scale={_aug_top_cam_focal_scale}, "
              f"arm_xy={_aug_arm_xy_shift}, arm_rz={_aug_arm_rot_z_deg}, "
              f"tbl_uv={_aug_table_uv_shift}, tbl_uv_rot={_aug_table_uv_rot_deg}, "
              f"cam_focal={_aug_cam_focal_jitter}, dome_rot={_aug_dome_rot_deg}, "
              f"cv2={'yes' if _aug_has_cv2 else 'no'})", flush=True)

        # --- Per-step garment color randomization ---
        # Every render step, fully randomize garment color via scale+bias on
        # the diffuseColor texture reader: output = texture * scale + bias.
        # Low scale washes out original hue, random bias injects a new color.
        # Only targets diffuseColorTex (skips roughness to preserve physical look).
        from pxr import Gf as _Gf_tint
        _color_debug_logged = [False]

        def _collect_garment_color_inputs_impl(env):
            """Find diffuseColor texture readers and cache scale+bias inputs."""
            _color_inputs.clear()
            _obj = env.object
            if _obj is None:
                return
            _stage = env.scene.stage
            # Cache the stage for the light augmentation hooks.
            _light_cache_stage_from_env(env)
            _targets, _ = _get_submesh_targets(_obj)
            _seen_mats = set()
            _sub_paths_all = []
            _do_debug = not _color_debug_logged[0]

            for _sub_path, _ in _targets:
                _sub_prim = _stage.GetPrimAtPath(_sub_path)
                _binding = _UsdShade_aug.MaterialBindingAPI(_sub_prim)
                _mat, _ = _binding.ComputeBoundMaterial()
                _sub_paths_all.append((_sub_path, _sub_prim))
                if not _mat:
                    continue
                _mat_path_str = str(_mat.GetPath())
                if _mat_path_str in _seen_mats:
                    continue
                _seen_mats.add(_mat_path_str)

                for _desc in _Usd_aug.PrimRange(_mat.GetPrim()):
                    if _desc.GetTypeName() != "Shader":
                        continue
                    _sh = _UsdShade_aug.Shader(_desc)
                    _sid = str(_sh.GetShaderId() or "")
                    if _do_debug:
                        _dbg_inputs = [_inp.GetBaseName() for _inp in _sh.GetInputs()]
                        print(f"[{label}] color debug: {_desc.GetPath()} id={_sid} inputs={_dbg_inputs}", flush=True)

                    # Only target diffuseColor texture readers (skip roughness etc.)
                    _node_name = _desc.GetName().lower()
                    _is_diffuse_tex = (
                        _sid == "UsdUVTexture"
                        and ("diffuse" in _node_name or "color" in _node_name)
                    )
                    if not _is_diffuse_tex:
                        continue

                    # Get or create scale input
                    _sc = _sh.GetInput("scale")
                    if not _sc:
                        try:
                            _sc = _sh.CreateInput("scale", _Sdf_aug.ValueTypeNames.Float4)
                            _sc.Set(_Gf_tint.Vec4f(1.0, 1.0, 1.0, 1.0))
                        except Exception as _e9:
                            if _do_debug:
                                print(f"[{label}] color: failed to create scale: {_e9}", flush=True)
                            _sc = None
                    # Get or create bias input
                    _bi = _sh.GetInput("bias")
                    if not _bi:
                        try:
                            _bi = _sh.CreateInput("bias", _Sdf_aug.ValueTypeNames.Float4)
                            _bi.Set(_Gf_tint.Vec4f(0.0, 0.0, 0.0, 0.0))
                        except Exception as _e9b:
                            if _do_debug:
                                print(f"[{label}] color: failed to create bias: {_e9b}", flush=True)
                            _bi = None
                    if _sc or _bi:
                        _color_inputs.append((_sc, _bi))
                        if _do_debug:
                            print(f"[{label}] color: cached scale+bias on {_desc.GetPath()}", flush=True)

            # Fallback: create flat-color material and bind it
            if not _color_inputs and _sub_paths_all:
                print(f"[{label}] color fallback: creating flat-color material", flush=True)
                _tint_mat = _UsdShade_aug.Material.Define(_stage, "/World/_step_tint_mat")
                _tint_sh = _UsdShade_aug.Shader.Define(_stage, "/World/_step_tint_mat/Shader")
                _tint_sh.CreateIdAttr("UsdPreviewSurface")
                _dc = _tint_sh.CreateInput("diffuseColor", _Sdf_aug.ValueTypeNames.Color3f)
                _dc.Set(_Gf_tint.Vec3f(1.0, 1.0, 1.0))
                _tint_mat.CreateSurfaceOutput().ConnectToSource(
                    _tint_sh.ConnectableAPI(), "surface"
                )
                for _sp, _sp_prim in _sub_paths_all:
                    _UsdShade_aug.MaterialBindingAPI(_sp_prim).Bind(_tint_mat)
                _color_inputs.append((None, _dc))  # special: (None, diffuseColor)
                print(f"[{label}] color fallback: bound to {len(_sub_paths_all)} submeshes", flush=True)

            _color_debug_logged[0] = True
            print(f"[{label}] color: collected {len(_color_inputs)} diffuse texture scale+bias pairs", flush=True)

        def _randomize_step_color_impl():
            """Randomize garment color: output = texture * scale + bias.

            scale ~ U(0.3, 0.8): preserve most of original pattern
            bias  ~ U(0.0, 0.35): gentle color shift
            Result: full color space, every step looks completely different.
            """
            _sr = _rng_aug.uniform(0.3, 0.8)
            _sg = _rng_aug.uniform(0.3, 0.8)
            _sb = _rng_aug.uniform(0.3, 0.8)
            _br = _rng_aug.uniform(0.0, 0.35)
            _bg = _rng_aug.uniform(0.0, 0.35)
            _bb = _rng_aug.uniform(0.0, 0.35)
            for _sc_inp, _bi_inp in _color_inputs:
                if _sc_inp is not None and _bi_inp is not None:
                    # Normal: texture reader with scale + bias
                    _sc_inp.Set(_Gf_tint.Vec4f(float(_sr), float(_sg), float(_sb), 1.0))
                    _bi_inp.Set(_Gf_tint.Vec4f(float(_br), float(_bg), float(_bb), 0.0))
                elif _sc_inp is None and _bi_inp is not None:
                    # Fallback flat-color material: just set diffuseColor
                    _bi_inp.Set(_Gf_tint.Vec3f(float(_br), float(_bg), float(_bb)))
                elif _sc_inp is not None:
                    # Only scale available
                    _sc_inp.Set(_Gf_tint.Vec4f(float(_br), float(_bg), float(_bb), 1.0))

        if _aug_step_color_tint:
            _collect_color_inputs = _collect_garment_color_inputs_impl
            _randomize_step_color = _randomize_step_color_impl
            print(f"[{label}] Per-step color tint enabled", flush=True)
        else:
            print(f"[{label}] Per-step color tint DISABLED", flush=True)

        # --- Per-step lighting ---
        # Uses env.scene.stage (cached after first episode init) for USD prim access.
        _DEFAULT_LIGHT_INTENSITY = 1200.0
        _DEFAULT_LIGHT_COLOR = (0.75, 0.75, 0.75)
        _DEFAULT_LIGHT_COLOR_TEMP = 6500.0  # neutral-ish daylight
        _light_prim_cache = [None]  # None = not yet searched, False = not found
        _light_temp_enabled = [False]  # set True once we've toggled enableColorTemperature
        _light_stage_cache = [None]

        def _light_get_stage():
            """Get USD stage from cached env reference."""
            if _light_stage_cache[0] is not None:
                return _light_stage_cache[0]
            try:
                from omni.usd import get_context
                _st = get_context().get_stage()
                if _st is not None:
                    _light_stage_cache[0] = _st
                return _st
            except Exception:
                return None

        def _light_cache_stage_from_env(env):
            """Cache stage from env (called once per episode from _collect_color_inputs)."""
            if _light_stage_cache[0] is None:
                try:
                    _light_stage_cache[0] = env.scene.stage
                except Exception:
                    pass

        def _init_light_cache():
            if _light_prim_cache[0] is not None:
                return
            _st = _light_get_stage()
            if _st is None:
                return
            _lp = _st.GetPrimAtPath("/World/Light")
            if _lp.IsValid():
                _light_prim_cache[0] = _lp
                print(f"[{label}] Light prim found at /World/Light", flush=True)
            else:
                _light_prim_cache[0] = False
                print(f"[{label}] Light prim NOT found at /World/Light", flush=True)

        def _randomize_step_light_impl():
            """Per-step light intensity / color / color-temperature randomization."""
            _init_light_cache()
            _lp = _light_prim_cache[0]
            if not _lp:
                return
            if _aug_light_intensity > 0:
                _i = _DEFAULT_LIGHT_INTENSITY + _rng_aug.uniform(-_aug_light_intensity, _aug_light_intensity)
                _lp.GetAttribute("inputs:intensity").Set(max(100.0, _i))
            if _aug_light_color > 0:
                _c = tuple(
                    max(0.0, min(1.0, _DEFAULT_LIGHT_COLOR[i] + _rng_aug.uniform(-_aug_light_color, _aug_light_color)))
                    for i in range(3)
                )
                _lp.GetAttribute("inputs:color").Set(_c)
            if _aug_light_color_temp > 0:
                if not _light_temp_enabled[0]:
                    try:
                        _en = _lp.GetAttribute("inputs:enableColorTemperature")
                        if not _en or not _en.IsValid():
                            _en = _lp.CreateAttribute("inputs:enableColorTemperature", _Sdf_aug.ValueTypeNames.Bool)
                        _en.Set(True)
                        _light_temp_enabled[0] = True
                    except Exception:
                        pass
                try:
                    _t = _DEFAULT_LIGHT_COLOR_TEMP + _rng_aug.uniform(-_aug_light_color_temp, _aug_light_color_temp)
                    _t = max(2000.0, min(12000.0, _t))
                    _ct = _lp.GetAttribute("inputs:colorTemperature")
                    if not _ct or not _ct.IsValid():
                        _ct = _lp.CreateAttribute("inputs:colorTemperature", _Sdf_aug.ValueTypeNames.Float)
                    _ct.Set(float(_t))
                except Exception:
                    pass

        if _aug_light_intensity > 0 or _aug_light_color > 0 or _aug_light_color_temp > 0:
            _randomize_step_light = _randomize_step_light_impl
            print(f"[{label}] Per-step light aug (intensity={_aug_light_intensity}, color={_aug_light_color}, color_temp={_aug_light_color_temp})", flush=True)

    return SimpleNamespace(
        enabled=enabled,
        apply_episode=_apply_garment_augmentation,
        collect_color_inputs=_collect_color_inputs,
        randomize_step_color=_randomize_step_color,
        randomize_step_light=_randomize_step_light,
        orig_roughness=_orig_roughness,
    )
