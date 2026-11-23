"""LIBERO+ observation/action adapter for the pinned LeRobot PI05 checkpoint."""
import io
import os
from contextlib import redirect_stdout

import numpy as np
import torch
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy


def axis_angle_xyzw(quat):
    """Convert LIBERO's xyzw EEF quaternion to a 3D axis-angle vector in radians."""
    q = np.asarray(quat, dtype=np.float32).copy()
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError(f"Invalid xyzw quaternion: {q}")
    q[3] = np.clip(q[3], -1.0, 1.0)
    den = np.sqrt(max(0.0, 1.0 - float(q[3]) ** 2))
    if den < 1e-10:
        return np.zeros(3, dtype=np.float32)
    return (q[:3] * (2.0 * np.arccos(q[3]) / den)).astype(np.float32)


class Adapter:
    image_size = 256

    def __init__(self):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        model_dir = os.environ["PI05_MODEL"]
        config = PI05Config.from_pretrained(model_dir)
        config.device = "cuda"
        config.compile_model = False  # Avoid expensive first-rollout compilation.
        config.n_action_steps = 10
        if config.chunk_size < config.n_action_steps:
            raise ValueError("PI05 checkpoint chunk is shorter than the rollout interval")
        log = io.StringIO()
        with redirect_stdout(log):
            self.policy = PI05Policy.from_pretrained(model_dir, config=config, strict=True).to("cuda").eval()
        load_log = log.getvalue()
        print(load_log[-5000:], flush=True)
        # This LeRobot version catches state-dict errors and can otherwise return
        # randomly initialized weights. Do not accept such a teacher silently.
        if "All keys loaded successfully!" not in load_log:
            raise RuntimeError("PI05 pretrained weights did not load strictly; aborting teacher rollout")
        self.preprocess, self.postprocess = make_pre_post_processors(
            self.policy.config,
            model_dir,
            preprocessor_overrides={"device_processor": {"device": "cuda"}},
        )
        state_dims = set()
        for step in self.preprocess.steps:
            stats = getattr(step, "stats", None) or {}
            for name in ("mean", "std"):
                value = stats.get("observation.state", {}).get(name)
                if value is not None:
                    state_dims.add(int(np.prod(np.shape(value))))
        if state_dims != {8}:
            raise RuntimeError(f"Expected saved 8D LIBERO state stats, found {state_dims}")
        print(f"PI05 loaded: {model_dir}; chunk={config.chunk_size}; "
              f"executed actions={config.n_action_steps}; state stats={state_dims}", flush=True)
        self.reset()

    def reset(self):
        self.policy.reset()
        for processor in (self.preprocess, self.postprocess):
            reset = getattr(processor, "reset", None)
            if callable(reset):
                reset()

    def observation_frame(self, obs, instruction):
        state = np.concatenate([
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            axis_angle_xyzw(obs["robot0_eef_quat"]),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
        ])
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError(f"Invalid LIBERO state: {state.shape}")
        frame = {"observation.state": torch.from_numpy(state), "task": instruction}
        for source, target in (
            ("agentview_image", "observation.images.image"),
            ("robot0_eye_in_hand_image", "observation.images.image2"),
        ):
            img = np.ascontiguousarray(obs[source][::-1, ::-1])
            if img.shape != (256, 256, 3) or img.dtype != np.uint8:
                raise ValueError(f"Unexpected {source}: {img.shape}, {img.dtype}")
            frame[target] = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return frame

    @torch.inference_mode()
    def predict(self, obs, instruction):
        batch = self.preprocess(self.observation_frame(obs, instruction))
        action = self.policy.select_action(batch)
        action = self.postprocess(action).detach().float().cpu().numpy()
        if action.shape != (1, 7) or not np.isfinite(action).all():
            raise ValueError(f"Invalid PI05 action: {action.shape}")
        return action.astype(np.float32, copy=False)
