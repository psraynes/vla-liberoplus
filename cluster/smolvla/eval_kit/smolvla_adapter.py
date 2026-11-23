"""Adapter for the configurable LIBERO+ kit; preserves SmolVLA's internal queue."""
import os
from contextlib import nullcontext

import numpy as np
import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors


def axis_angle_xyzw(quat):
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
            raise RuntimeError("CUDA is unavailable in this Python environment.")
        model_id = os.environ.get("SMOLVLA_MODEL", "lerobot/smolvla_libero")
        self.policy = SmolVLAPolicy.from_pretrained(model_id).to("cuda").eval()
        self.policy.config.device = "cuda"
        self.policy.config.n_action_steps = int(os.environ.get("SMOLVLA_ACTION_STEPS", "10"))
        if not 1 <= self.policy.config.n_action_steps <= self.policy.config.chunk_size:
            raise ValueError("Action steps must be between 1 and the generated chunk size.")
        self.preprocess, self.postprocess = make_pre_post_processors(
            self.policy.config,
            model_id,
            preprocessor_overrides={"device_processor": {"device": "cuda"}},
        )
        # The published config declares 6 state values; actual saved stats must
        # match the 8D state used by LeRobot's LIBERO evaluator. Never truncate it.
        state_dims = set()
        for step in self.preprocess.steps:
            stats = getattr(step, "stats", None) or {}
            state_stats = stats.get("observation.state", {})
            for name in ("mean", "std"):
                if name in state_stats:
                    state_dims.add(int(np.prod(np.shape(state_stats[name]))))
        if state_dims != {8}:
            raise RuntimeError(
                f"Expected saved LIBERO state mean/std dimension 8; found {state_dims}. "
                "Use the exact checkpoint/processors from the working clean run; "
                "do not truncate or replace normalization statistics."
            )
        print(f"SmolVLA loaded: {model_id}; generated chunk={self.policy.config.chunk_size}; "
              f"executed actions={self.policy.config.n_action_steps}; state stats={state_dims}", flush=True)
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
            raise ValueError(f"Invalid LIBERO state: shape={state.shape}")
        frame = {"observation.state": torch.from_numpy(state), "task": instruction}
        for raw, key in (
            ("agentview_image", "observation.images.camera1"),
            ("robot0_eye_in_hand_image", "observation.images.camera2"),
        ):
            img = np.ascontiguousarray(obs[raw][::-1, ::-1])
            if img.shape != (256, 256, 3) or img.dtype != np.uint8:
                raise ValueError(f"Unexpected {raw}: {img.shape}, {img.dtype}")
            frame[key] = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return frame

    @torch.inference_mode()
    def predict(self, obs, instruction):
        batch = self.preprocess(self.observation_frame(obs, instruction))
        if batch["observation.state"].shape != (1, 8):
            raise ValueError("Processor did not preserve batched 8D LIBERO state.")
        amp = (torch.autocast("cuda") if self.policy.config.use_amp else nullcontext())
        with amp:
            action = self.policy.select_action(batch)
        action = self.postprocess(action).detach().float().cpu().numpy()
        if action.shape != (1, 7) or not np.isfinite(action).all():
            raise ValueError(f"Invalid action: {action.shape}")
        # select_action caches 10 actions internally; returning one row avoids
        # introducing a second queue or generating a fresh chunk every step.
        # LeRobot sends the unnormalized action directly to LIBERO: no sign flip.
        return action.astype(np.float32, copy=False)
