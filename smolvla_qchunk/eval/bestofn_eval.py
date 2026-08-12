"""Utilities to evaluate SmolVLA with an optional Best-of-N critic."""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass, fields
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any, Callable, Optional

import draccus
import torch
from huggingface_hub.constants import CONFIG_NAME

from lerobot.configs.types import FeatureType
from lerobot.envs.configs import EnvConfig
from lerobot.envs.factory import make_env
from lerobot.envs.utils import close_envs, preprocess_observation
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.utils import populate_queues
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.constants import ACTION
from lerobot.utils.random_utils import set_seed

from qchunk.best_of_n_critic import BestOfNCriticTrainer
from qchunk.qchunked_critic import QChunkedCritic
from scripts.train_qchunk_offline import CriticConfig, encode_policy_observations
from utils.local_policy import make_local_smolvla_policy

CriticRuntime = BestOfNCriticTrainer | QChunkedCritic


@dataclass
class CriticCheckpoint:
    state_dict: dict[str, Any]
    config: CriticConfig
    raw_config: dict[str, Any] | None
    meta: dict[str, Any]


@dataclass
class PolicyEvalBundle:
    policy: Any
    preprocessor: Any
    postprocessor: Any
    critic_trainer: Optional[CriticRuntime]
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
    n_action_steps: int = 10,
    policy_num_steps: int | None = None,
    critic_q_agg_override: str | None = None,
    use_vlm_backbone_encode: bool = True,
    policy_bundle: PolicyEvalBundle | None = None,
    use_current_critic: bool = False,
) -> dict[str, Any]:
    """Run evaluation rollouts with optional critic-guided action selection."""

    set_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    envs = make_env(env_cfg, n_envs=batch_size, use_async_envs=False)

    if policy_bundle is None:
        critic_cfg_dict = _peek_critic_raw_config(critic_state_path)
        effective_use_vlm_backbone_encode = _effective_use_vlm_backbone_encode(
            critic_cfg_dict,
            fallback=use_vlm_backbone_encode,
        )
        encoder_fn = lambda p, b: encode_policy_observations(
            p,
            b,
            use_vlm_backbone_encode=effective_use_vlm_backbone_encode,
        )
        policy_cfg = _load_policy_config(policy_path)
        policy_cfg.device = device
        policy_cfg.n_action_steps = n_action_steps
        if policy_num_steps is not None:
            policy_cfg.num_steps = int(policy_num_steps)
        policy = make_local_smolvla_policy(cfg=policy_cfg, env_cfg=env_cfg)
        policy.eval()

        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=str(policy_path),
            preprocessor_overrides={"device_processor": {"device": device}},
        )

        critic_trainer = None
        if critic_state_path is not None:
            critic_trainer = _build_eval_critic(
                policy_path=policy_path,
                policy=policy,
                envs=envs,
                preprocessor=preprocessor,
                critic_ckpt_path=critic_state_path,
                device=device,
                best_of_n=best_of_n,
                critic_q_agg_override=critic_q_agg_override,
                encoder_fn=encoder_fn,
                effective_use_vlm_backbone_encode=effective_use_vlm_backbone_encode,
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
        if best_of_n > 1 and critic_state_path is not None and critic_trainer is None:
            logging.warning("Best-of-N requested but critic checkpoint was not found. Falling back to BC.")
        if best_of_n > 1 and critic_state_path is None and critic_trainer is None:
            logging.warning("Best-of-N requested but no critic checkpoint provided. Falling back to BC.")
    if n_action_steps is None:
        raise ValueError("n_action_steps must be provided for evaluation.")
    needs_critic_patch = best_of_n > 1 and critic_trainer is not None
    already_patched = getattr(policy, "_best_of_n_patched", False)
    if needs_critic_patch and not already_patched:
        _patch_policy_select_action(
            policy,
            critic_trainer if needs_critic_patch else None,
            encoder_fn,
            best_of_n=best_of_n,
            n_action_steps=n_action_steps,
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
    info["n_action_steps"] = getattr(policy_config, "n_action_steps", None)
    info["policy_num_steps"] = getattr(policy_config, "num_steps", None)

    close_envs(envs)
    return info


def _patch_policy_select_action(
    policy: Any,
    critic_trainer: CriticRuntime | None,
    encoder_fn: Callable[[Any, dict[str, torch.Tensor]], Any],
    *,
    best_of_n: int,
    n_action_steps: int,
) -> None:
    """Replace the policy select_action with a critic-guided version."""

    def select_action_with_critic(self, batch: dict[str, torch.Tensor], noise=None):
        self.eval()
        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])
        if len(self._queues[ACTION]) == 0:
            if critic_trainer is not None:
                if isinstance(critic_trainer, QChunkedCritic):
                    actions, _ = critic_trainer.predict_best_of_n(
                        batch,
                        action_samples=best_of_n,
                        actor=self,
                    )
                else:
                    encoding = encoder_fn(self, batch)
                    actions, _ = critic_trainer._best_of_n_actions_soon(self, batch, encoding)
            else:
                actions = self.predict_action_chunk(batch)
            per_step_actions = actions.transpose(0, 1)
            exec_steps = min(n_action_steps, per_step_actions.shape[0])
            self._queues[ACTION].extend(per_step_actions[:exec_steps])
        return self._queues[ACTION].popleft()

    policy.select_action = MethodType(select_action_with_critic, policy)
    setattr(policy, "_best_of_n_patched", True)


def _build_eval_critic(
    *,
    policy_path: Path,
    policy: Any,
    envs,
    preprocessor,
    critic_ckpt_path: Path,
    device: str,
    best_of_n: int,
    critic_q_agg_override: str | None = None,
    encoder_fn: Callable[[Any, dict[str, torch.Tensor]], Any],
    effective_use_vlm_backbone_encode: bool,
    use_current_critic: bool = False,
) -> CriticRuntime | None:
    checkpoint_file = _resolve_critic_state_path(critic_ckpt_path)
    if checkpoint_file is None:
        return None
    ckpt = _load_critic_checkpoint(checkpoint_file)
    sample_batch, sample_size = _sample_policy_batch(envs, preprocessor, device, policy)
    builder_batch = dict(sample_batch)
    builder_batch["action"] = _dummy_action_tensor(policy, sample_size, device)
    if _needs_qchunk_runtime(ckpt.raw_config):
        runtime_cfg = SimpleNamespace(**(ckpt.raw_config or {}))
        runtime_cfg.action_samples = max(best_of_n, 1)
        runtime_cfg.use_vlm_backbone_encode = effective_use_vlm_backbone_encode
        if critic_q_agg_override is not None:
            runtime_cfg.q_aggregation = critic_q_agg_override
        critic = QChunkedCritic.build(
            policy_path=policy_path,
            policy_cfg=policy.config,
            critic_cfg=runtime_cfg,
            sample_batch=builder_batch,
            ds_meta=None,
            device=device,
            encoder_fn=encoder_fn,
            actor=policy,
            freeze_actor=True,
        )
        critic.load_state_dict(ckpt.state_dict)
        logging.info(
            "[QChunk eval] critic loaded from %s | cse=%s indep=%s vlm_enc=%s "
            "state_encoder=%s vision_encoder=%s lang_embedding=%s",
            checkpoint_file,
            critic.use_critic_state_encoder,
            critic.use_independent_critic_encoder,
            getattr(runtime_cfg, "use_vlm_backbone_encode", None),
            "loaded" if critic.state_encoder is not None else "None",
            "loaded" if critic.critic_vision_encoder is not None else "None",
            "loaded" if critic.critic_lang_embedding is not None else "None",
        )
        if use_current_critic:
            _use_online_qchunk_modules_for_eval(critic)
        critic.critic.eval()
        critic.target_critic.eval()
        if critic.state_encoder is not None:
            critic.state_encoder.eval()
        if critic.target_state_encoder is not None:
            critic.target_state_encoder.eval()
        if critic.critic_vision_encoder is not None:
            critic.critic_vision_encoder.eval()
        if critic.target_critic_vision_encoder is not None:
            critic.target_critic_vision_encoder.eval()
        if critic.vision_proj is not None:
            critic.vision_proj.eval()
        if critic.target_vision_proj is not None:
            critic.target_vision_proj.eval()
        if critic.lang_proj is not None:
            critic.lang_proj.eval()
        if critic.target_lang_proj is not None:
            critic.target_lang_proj.eval()
        if critic.critic_lang_embedding is not None:
            critic.critic_lang_embedding.eval()
        return critic
    ckpt.config.action_samples = max(best_of_n, 1)
    if critic_q_agg_override is not None:
        ckpt.config.q_aggregation = critic_q_agg_override
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
    logging.info(
        "[Legacy eval] critic loaded from %s | critic_type=%s q_agg=%s (BestOfNCriticTrainer path, no state/vision encoder)",
        checkpoint_file,
        getattr(ckpt.config, "critic_type", None),
        getattr(ckpt.config, "q_aggregation", None),
    )
    # Evaluation only: reuse the online critic directly instead of the (stale) target copy
    if use_current_critic:
        # print("using current critic for eval")
        trainer.target_critic = trainer.critic
    trainer.critic.eval()
    trainer.target_critic.eval()
    return trainer


def _dummy_action_tensor(policy: Any, batch_size: int, device: str) -> torch.Tensor:
    chunk_size = getattr(policy.config, "chunk_size", 1)
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


def _normalize_critic_config_payload(cfg_dict: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(cfg_dict, dict):
        return None

    def _convert(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {k: _convert(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_convert(v) for v in value]
        if isinstance(value, tuple):
            return tuple(_convert(v) for v in value)
        return value

    normalized = _convert(cfg_dict)
    if "ood_source" in normalized and "ood_action_source" not in normalized:
        normalized["ood_action_source"] = normalized["ood_source"]
    if "vqh_num_backbone_layers" in normalized and "qformer_num_backbone_layers" not in normalized:
        normalized["qformer_num_backbone_layers"] = normalized["vqh_num_backbone_layers"]
    if "cql_m_actions" in normalized and "ood_m_actions" not in normalized:
        normalized["ood_m_actions"] = normalized["cql_m_actions"]
    if "pairwise_ood_loss_weight" in normalized and "loss_rank_weight" not in normalized:
        normalized["loss_rank_weight"] = normalized["pairwise_ood_loss_weight"]
    if "use_pairwise_ood_loss" in normalized:
        if not normalized["use_pairwise_ood_loss"]:
            normalized["loss_rank_weight"] = 0.0
        normalized.pop("use_pairwise_ood_loss", None)

    return normalized


def _normalize_critic_config(cfg_dict: dict[str, Any] | None) -> dict[str, Any] | None:
    normalized = _normalize_critic_config_payload(cfg_dict)
    if normalized is None:
        return None
    allowed = set(CriticConfig.__dataclass_fields__.keys())
    return {key: value for key, value in normalized.items() if key in allowed}


def _load_critic_config_file(checkpoint_path: Path) -> dict[str, Any] | None:
    candidates = []
    if checkpoint_path.is_file():
        candidates.append(checkpoint_path.parent / "config.json")
    if checkpoint_path.is_dir():
        candidates.append(checkpoint_path / "config.json")
        candidates.append(checkpoint_path / "critic_pretrained_model" / "config.json")
    for candidate in candidates:
        if candidate.exists():
            with candidate.open("r", encoding="utf-8") as f:
                return json.load(f)
    return None


def _load_critic_checkpoint(path: Path) -> CriticCheckpoint:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config_payload = _load_critic_config_file(path)
    if "state_dict" in payload:
        state_dict = payload["state_dict"]
        cfg_dict = config_payload if config_payload is not None else payload.get("critic_config")
        meta = payload.get("meta", {})
    else:
        state_dict = payload
        cfg_dict = config_payload
        meta = {}
    raw_cfg = _normalize_critic_config_payload(cfg_dict)
    normalized_cfg = _normalize_critic_config(raw_cfg)
    cfg = CriticConfig(**normalized_cfg) if normalized_cfg is not None else CriticConfig()
    return CriticCheckpoint(state_dict=state_dict, config=cfg, raw_config=raw_cfg, meta=meta)


def _resolve_critic_state_path(path: Path, *, warn: bool = True) -> Path | None:
    if path.is_file():
        return path
    if path.is_dir():
        candidate = path / "critic_pretrained_model" / "last.ckpt"
        if candidate.exists():
            return candidate
        candidate = path / "last.ckpt"
        if candidate.exists():
            return candidate
        alternate = path / "critic_state.pt"
        if alternate.exists():
            return alternate
    if warn:
        logging.warning("Could not find critic checkpoint at %s", path)
    return None


def _peek_critic_raw_config(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    checkpoint_path = _resolve_critic_state_path(path, warn=False)
    if checkpoint_path is None:
        return None
    return _normalize_critic_config_payload(_load_critic_config_file(checkpoint_path))


def _effective_use_vlm_backbone_encode(cfg_dict: dict[str, Any] | None, *, fallback: bool) -> bool:
    if not isinstance(cfg_dict, dict):
        return bool(fallback)
    return bool(cfg_dict.get("use_vlm_backbone_encode", fallback))


def _needs_qchunk_runtime(cfg_dict: dict[str, Any] | None) -> bool:
    if not isinstance(cfg_dict, dict):
        return False
    return bool(
        cfg_dict.get("use_critic_state_encoder")
        or cfg_dict.get("use_independent_critic_encoder")
    )


def _use_online_qchunk_modules_for_eval(critic: QChunkedCritic) -> None:
    critic.target_critic = critic.critic
    if critic.state_encoder is not None:
        critic.target_state_encoder = critic.state_encoder
    if critic.critic_vision_encoder is not None:
        critic.target_critic_vision_encoder = critic.critic_vision_encoder
    if critic.vision_proj is not None:
        critic.target_vision_proj = critic.vision_proj
    if critic.lang_proj is not None:
        critic.target_lang_proj = critic.lang_proj


def build_policy_eval_bundle(
    *,
    policy_path: Path,
    critic_state_path: Path | None,
    sample_env_cfg: EnvConfig,
    sample_batch_size: int,
    device: str,
    best_of_n: int = 1,
    n_action_steps: int = 10,
    policy_num_steps: int | None = None,
    critic_q_agg_override: str | None = None,
    use_vlm_backbone_encode: bool = True,
    use_current_critic: bool = False,
) -> PolicyEvalBundle:
    """Load policy/critic once and build reusable components for multiple tasks."""
    # print(policy_path)
    policy_cfg = _load_policy_config(policy_path)
    policy_cfg.device = device
    policy_cfg.n_action_steps = n_action_steps
    if policy_num_steps is not None:
        policy_cfg.num_steps = int(policy_num_steps)
    policy = make_local_smolvla_policy(cfg=policy_cfg, env_cfg=sample_env_cfg)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(policy_path),
        preprocessor_overrides={"device_processor": {"device": device}},
    )

    envs = make_env(sample_env_cfg, n_envs=sample_batch_size, use_async_envs=False)
    critic_cfg_dict = _peek_critic_raw_config(critic_state_path)
    effective_use_vlm_backbone_encode = _effective_use_vlm_backbone_encode(
        critic_cfg_dict,
        fallback=use_vlm_backbone_encode,
    )

    encoder_fn = lambda p, b: encode_policy_observations(
        p,
        b,
        use_vlm_backbone_encode=effective_use_vlm_backbone_encode,
    )
    critic_trainer = None
    if n_action_steps is None:
        raise ValueError("n_action_steps must be provided for evaluation.")
    if critic_state_path is not None:
        critic_trainer = _build_eval_critic(
            policy_path=policy_path,
            policy=policy,
            envs=envs,
            preprocessor=preprocessor,
            critic_ckpt_path=critic_state_path,
            device=device,
            best_of_n=best_of_n,
            critic_q_agg_override=critic_q_agg_override,
            encoder_fn=encoder_fn,
            effective_use_vlm_backbone_encode=effective_use_vlm_backbone_encode,
            use_current_critic=use_current_critic,
        )
        if critic_trainer is not None and best_of_n > 1:
            _patch_policy_select_action(
                policy,
                critic_trainer,
                encoder_fn,
                best_of_n=best_of_n,
                n_action_steps=n_action_steps,
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
    valid_fields = {field.name for field in fields(SmolVLAConfig)}
    unknown_fields = sorted(key for key in payload.keys() if key not in valid_fields)
    if unknown_fields:
        logging.warning(
            "Dropping unsupported SmolVLA config fields for current lerobot version: %s",
            ", ".join(unknown_fields),
        )
        payload = {key: value for key, value in payload.items() if key in valid_fields}
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
    # MetaWorld policies may require both observation.state and
    # observation.environment_state; mirror state when env_state is absent.
    if "observation.state" in observation and "observation.environment_state" not in observation:
        observation["observation.environment_state"] = observation["observation.state"]

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
