"""Utilities to evaluate SmolVLA with an optional Best-of-N critic."""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Any, Callable, Dict, Optional

import draccus
import torch
from huggingface_hub.constants import CONFIG_NAME

from lerobot.configs.types import FeatureType
from lerobot.envs.configs import EnvConfig, LiberoEnv
from lerobot.envs.factory import make_env
from lerobot.envs.utils import add_envs_task, close_envs, preprocess_observation
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.utils import populate_queues
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.constants import ACTION
from lerobot.utils.random_utils import set_seed

from qchunk.best_of_n_critic import BestOfNCriticTrainer
from scripts.train_qchunk_offline import CriticConfig, encode_policy_observations_test


@dataclass
class CriticCheckpoint:
    state_dict: dict[str, Any]
    config: CriticConfig
    meta: dict[str, Any]


@dataclass
class PolicyEvalBundle:
    policy: Any
    preprocessor: Any
    postprocessor: Any
    critic_trainer: Optional[BestOfNCriticTrainer]
    use_amp: bool
    encoder_fn: Callable[[Any, dict[str, torch.Tensor]], Any]


def evaluate_policy_with_best_of_n(
    *,
    policy_path: Path,
    critic_state_path: Path | None,
    env_cfg: EnvConfig,
    n_episodes: int,
    batch_size: int,
    seed: int,
    device: str,
    best_of_n: int = 1,
    videos_dir: Path | None = None,
    max_render: int = 0,
    policy_n_action_steps: int = 10,
    exec_n_action_steps: int | None = None,
    critic_q_agg_override: str | None = None,
    use_data_augmentations: bool = False,
    use_vlm_backbone_encode: bool = True,
    policy_bundle: PolicyEvalBundle | None = None,
    use_current_critic: bool = False,
) -> dict[str, Any]:
    """Run evaluation rollouts with optional critic-guided action selection."""

    set_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    envs = make_env(env_cfg, n_envs=batch_size, use_async_envs=False)

    encoder_fn = (
        policy_bundle.encoder_fn
        if policy_bundle is not None
        else lambda p, b: encode_policy_observations_test(
            p,
            b,
            use_data_augmentations=use_data_augmentations,
            use_vlm_backbone_encode=use_vlm_backbone_encode,
        )
    )

    if policy_bundle is None:
        policy_cfg = _load_policy_config(policy_path)
        policy_cfg.device = device
        policy_cfg.n_action_steps = policy_n_action_steps
        policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
        ########################
        print(f"policy_cfg.n_action_steps:  {policy_cfg.n_action_steps}")
        ########################
        policy.eval()

        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=str(policy_path),
            preprocessor_overrides={"device_processor": {"device": device}},
        )

        critic_trainer = None
        exec_action_steps = exec_n_action_steps or policy_n_action_steps
        if critic_state_path is not None:
            critic_trainer = _build_eval_critic(
                policy=policy,
                envs=envs,
                preprocessor=preprocessor,
                critic_ckpt_path=critic_state_path,
                device=device,
                best_of_n=best_of_n,
                critic_q_agg_override=critic_q_agg_override,
                encoder_fn=encoder_fn,
                use_current_critic=use_current_critic,
            )
            if critic_trainer is None and best_of_n > 1:
                logging.warning("Best-of-N requested but critic checkpoint was not found. Falling back to BC.")
        elif best_of_n > 1:
            logging.warning("Best-of-N requested but no critic checkpoint provided. Falling back to BC.")
        use_amp = getattr(policy_cfg, "use_amp", False)
    else:
        policy = policy_bundle.policy
        preprocessor = policy_bundle.preprocessor
        postprocessor = policy_bundle.postprocessor
        critic_trainer = policy_bundle.critic_trainer
        use_amp = policy_bundle.use_amp
        encoder_fn = policy_bundle.encoder_fn
        policy.reset()
        policy.eval()
        exec_action_steps = exec_n_action_steps or policy_n_action_steps
        if best_of_n > 1 and critic_state_path is not None and critic_trainer is None:
            logging.warning("Best-of-N requested but critic checkpoint was not found. Falling back to BC.")
        if best_of_n > 1 and critic_state_path is None and critic_trainer is None:
            logging.warning("Best-of-N requested but no critic checkpoint provided. Falling back to BC.")
    if exec_action_steps is None:
        raise ValueError("exec_n_action_steps or policy_n_action_steps must be provided for evaluation.")
    needs_critic_patch = best_of_n > 1 and critic_trainer is not None
    needs_exec_patch = exec_action_steps != policy_n_action_steps
    already_patched = getattr(policy, "_best_of_n_patched", False)
    prev_exec_steps = getattr(policy, "_exec_n_action_steps", None)
    if (needs_critic_patch or needs_exec_patch) and (not already_patched or prev_exec_steps != exec_action_steps):
        _patch_policy_select_action(
            policy,
            critic_trainer if needs_critic_patch else None,
            encoder_fn,
            exec_n_action_steps=exec_action_steps,
        )

    videos_path = None
    if videos_dir is not None:
        videos_path = videos_dir
        videos_path.mkdir(parents=True, exist_ok=True)

    device_type = torch.device(device).type
    with torch.no_grad(), torch.autocast(device_type=device_type, enabled=use_amp):
        info = eval_policy_all(
            envs=envs,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=n_episodes,
            max_episodes_rendered=max_render,
            videos_dir=videos_path,
            start_seed=seed,
            max_parallel_tasks=getattr(env_cfg, "max_parallel_tasks", batch_size),
        )
    policy_config = getattr(policy, "config", None)
    info["policy_n_action_steps"] = getattr(policy_config, "n_action_steps", None)
    info["exec_n_action_steps"] = exec_action_steps

    close_envs(envs)
    return info


def _patch_policy_select_action(
    policy: Any,
    critic_trainer: BestOfNCriticTrainer | None,
    encoder_fn: Callable[[Any, dict[str, torch.Tensor]], Any],
    *,
    exec_n_action_steps: int,
) -> None:
    """Replace the policy select_action with a critic-guided version."""

    def select_action_with_critic(self, batch: dict[str, torch.Tensor], noise=None):
        self.eval()
        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])
        if len(self._queues[ACTION]) == 0:
            if critic_trainer is not None:
                encoding = encoder_fn(self, batch)
                actions, _ = critic_trainer._best_of_n_actions_soon(self, batch, encoding)
            else:
                actions = self.predict_action_chunk(batch)
            per_step_actions = actions.transpose(0, 1)
            exec_steps = min(exec_n_action_steps, per_step_actions.shape[0])
            if exec_steps < exec_n_action_steps:
                logging.warning(
                    "Requested exec_n_action_steps=%s exceeds available chunk=%s; truncating.",
                    exec_n_action_steps,
                    per_step_actions.shape[0],
                )
            self._queues[ACTION].extend(per_step_actions[:exec_steps])
        return self._queues[ACTION].popleft()

    policy.select_action = MethodType(select_action_with_critic, policy)
    setattr(policy, "_best_of_n_patched", True)
    setattr(policy, "_exec_n_action_steps", exec_n_action_steps)


def _build_eval_critic(
    *,
    policy: Any,
    envs,
    preprocessor,
    critic_ckpt_path: Path,
    device: str,
    best_of_n: int,
    critic_q_agg_override: str | None = None,
    encoder_fn: Callable[[Any, dict[str, torch.Tensor]], Any],
    use_current_critic: bool = False,
) -> BestOfNCriticTrainer | None:
    checkpoint_file = _resolve_critic_state_path(critic_ckpt_path)
    if checkpoint_file is None:
        return None
    ckpt = _load_critic_checkpoint(checkpoint_file)
    ckpt.config.action_samples = max(best_of_n, 1)
    if critic_q_agg_override is not None:
        ckpt.config.q_aggregation = critic_q_agg_override

    sample_batch, sample_size = _sample_policy_batch(envs, preprocessor, device, policy)
    builder_batch = dict(sample_batch)
    builder_batch["action"] = _dummy_action_tensor(policy, sample_size, device)
    trainer = BestOfNCriticTrainer.build(
        policy=policy,
        batch=builder_batch,
        cfg=ckpt.config,
        device=torch.device(device),
        encoder_fn=encoder_fn,
    )
    if type(trainer.critic) != type(trainer.target_critic):
        raise TypeError("Critic type mismatch; ensure evaluation config matches training critic_type.")
    trainer.load_state_dict(ckpt.state_dict)
    # Evaluation only: reuse the online critic directly instead of the (stale) target copy
    if use_current_critic:
        print("using current critic for eval")
        trainer.target_critic = trainer.critic
    trainer.critic.eval()
    trainer.target_critic.eval()
    return trainer


def _dummy_action_tensor(policy: Any, batch_size: int, device: str) -> torch.Tensor:
    chunk_size = getattr(policy.config, "n_action_steps", 1)
    action_dim = _infer_action_dim(policy)
    return torch.zeros(batch_size, chunk_size, action_dim, device=device)


def _infer_batch_size(batch: dict[str, Any]) -> int:
    for value in batch.values():
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            return value.shape[0]
        if isinstance(value, dict) and value:
            try:
                size = _infer_batch_size(value)
            except ValueError:
                continue
            if size is not None:
                return size
    raise ValueError("Unable to infer batch size from preprocessed observation.")


def _infer_action_dim(policy: Any) -> int:
    output_features = getattr(policy.config, "output_features", {})
    for feature in output_features.values():
        if feature.type is FeatureType.ACTION:
            return feature.shape[-1]
    raise ValueError("Could not infer action dimension from policy configuration.")


def _sample_policy_batch(envs, preprocessor, device: str, policy: Any) -> tuple[dict[str, torch.Tensor], int]:
    suite_name = next(iter(envs))
    env_map = envs[suite_name]
    first_task = next(iter(env_map))
    vec_env = env_map[first_task]
    observation, _ = vec_env.reset()
    observation = preprocess_observation(observation)
    observation = _augment_with_task(vec_env, observation)
    policy_ready = policy._prepare_batch(observation)
    batch = preprocessor(policy_ready)
    batch = _to_device(batch, device)
    policy.reset()
    if hasattr(vec_env, "num_envs"):
        batch_size = vec_env.num_envs
    else:
        try:
            batch_size = _infer_batch_size(batch)
        except ValueError:
            batch_size = 1
    return batch, batch_size


def _to_device(payload: dict[str, Any], device: str) -> dict[str, Any]:
    converted: dict[str, Any] = {}
    for key, value in payload.items():
        converted[key] = _maybe_to_device(value, device)
    return converted


def _maybe_to_device(value: Any, device: str):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return _to_device(value, device)
    if isinstance(value, list):
        return [_maybe_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_maybe_to_device(item, device) for item in value)
    return value


def _load_critic_checkpoint(path: Path) -> CriticCheckpoint:
    payload = torch.load(path, map_location="cpu")
    if "state_dict" in payload:
        state_dict = payload["state_dict"]
        cfg_dict = payload.get("critic_config")
        meta = payload.get("meta", {})
    else:
        state_dict = payload
        cfg_dict = None
        meta = {}
    cfg = CriticConfig(**cfg_dict) if isinstance(cfg_dict, dict) else CriticConfig()
    return CriticCheckpoint(state_dict=state_dict, config=cfg, meta=meta)


def _resolve_critic_state_path(path: Path) -> Path | None:
    if path.is_file():
        return path
    if path.is_dir():
        candidate = path / "critic_pretrained_model" / "last.ckpt"
        if candidate.exists():
            return candidate
        alternate = path / "critic_state.pt"
        if alternate.exists():
            return alternate
    logging.warning("Could not find critic checkpoint at %s", path)
    return None


def build_policy_eval_bundle(
    *,
    policy_path: Path,
    critic_state_path: Path | None,
    sample_env_cfg: EnvConfig,
    sample_batch_size: int,
    device: str,
    best_of_n: int = 1,
    policy_n_action_steps: int = 10,
    exec_n_action_steps: int | None = None,
    critic_q_agg_override: str | None = None,
    use_data_augmentations: bool = False,
    use_vlm_backbone_encode: bool = True,
    use_current_critic: bool = False,
) -> PolicyEvalBundle:
    """Load policy/critic once and build reusable components for multiple tasks."""
    print(policy_path)
    policy_cfg = _load_policy_config(policy_path)
    policy_cfg.device = device
    policy_cfg.n_action_steps = policy_n_action_steps
    policy = make_policy(cfg=policy_cfg, env_cfg=sample_env_cfg)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(policy_path),
        preprocessor_overrides={"device_processor": {"device": device}},
    )

    envs = make_env(sample_env_cfg, n_envs=sample_batch_size, use_async_envs=False)

    encoder_fn = lambda p, b: encode_policy_observations_test(
        p,
        b,
        use_data_augmentations=use_data_augmentations,
        use_vlm_backbone_encode=use_vlm_backbone_encode,
    )
    critic_trainer = None
    exec_action_steps = exec_n_action_steps or policy_n_action_steps
    if exec_action_steps is None:
        raise ValueError("exec_n_action_steps or policy_n_action_steps must be provided for evaluation.")
    if critic_state_path is not None:
        critic_trainer = _build_eval_critic(
            policy=policy,
            envs=envs,
            preprocessor=preprocessor,
            critic_ckpt_path=critic_state_path,
            device=device,
            best_of_n=best_of_n,
            critic_q_agg_override=critic_q_agg_override,
            encoder_fn=encoder_fn,
            use_current_critic=use_current_critic,
        )
        if critic_trainer is not None and best_of_n > 1:
            _patch_policy_select_action(
                policy,
                critic_trainer,
                encoder_fn,
                exec_n_action_steps=exec_action_steps,
            )
    close_envs(envs)

    use_amp = getattr(policy_cfg, "use_amp", False)
    return PolicyEvalBundle(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        critic_trainer=critic_trainer,
        use_amp=use_amp,
        encoder_fn=encoder_fn,
    )


def _load_policy_config(policy_path: Path) -> SmolVLAConfig:
    config_path = policy_path / CONFIG_NAME
    if not config_path.exists():
        raise FileNotFoundError(f"Could not find config.json under {policy_path}.")
    with config_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    payload.pop("type", None)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as tmp:
            json.dump(payload, tmp)
            tmp_path = Path(tmp.name)
        with draccus.config_type("json"):
            cfg = draccus.parse(SmolVLAConfig, str(tmp_path), args=[])
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()
    cfg.pretrained_path = str(policy_path)
    return cfg


def _augment_with_task(vec_env: Any, observation: dict[str, Any]) -> dict[str, Any]:
    base_envs = getattr(vec_env, "envs", None)
    if base_envs and hasattr(base_envs[0], "task_description"):
        task_result = vec_env.call("task_description")
        if isinstance(task_result, tuple):
            task_result = list(task_result)
        if not isinstance(task_result, list) or not all(isinstance(item, str) for item in task_result):
            raise TypeError("task_description must return a list of strings.")
        observation["task"] = task_result
    elif base_envs and hasattr(base_envs[0], "task"):
        task_result = vec_env.call("task")
        if isinstance(task_result, tuple):
            task_result = list(task_result)
        if not isinstance(task_result, list) or not all(isinstance(item, str) for item in task_result):
            raise TypeError("task must return a list of strings.")
        observation["task"] = task_result
    else:
        num_envs = next(iter(observation.values())).shape[0]
        observation["task"] = ["" for _ in range(num_envs)]
    return observation
