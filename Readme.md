# VGAS: Value-Guided Action-Chunk Selection and Refinement for Few-Shot VLA Adaptation

[![arXiv](https://img.shields.io/badge/arXiv-2602.07399-b31b1b.svg)](https://arxiv.org/abs/2602.07399)
[![Project Page](https://img.shields.io/badge/Project-Page-1f6feb.svg)](https://jyugo-15.github.io/VGAS/)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Checkpoints-ffce3a.svg)](https://huggingface.co/SemyonXu616/VGAS-5-shot)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-5--shot%20LIBERO-ffce3a.svg)](https://huggingface.co/datasets/SemyonXu616/HF_LIBERO_5_SHOT)

Official implementation of **VGAS** and **VGAS+**, a generation–selection
framework for few-shot Vision-Language-Action (VLA) adaptation.

Under scarce demonstrations, a fine-tuned VLA policy generally produces semantically plausible
action chunks, but small geometric errors can flip the execution outcome from success to failure.
We address this by learning a long-horizon critic over temporally extended action chunks:

- **Q-Chunk-Former** — a geometrically grounded Transformer critic that maps vision–language
  observations and proprioceptive state to chunk-level values.
- **Explicit Geometric Regularization (EGR)** — injects dense geometric supervision around expert
  demonstrations to preserve fine-grained value discrimination among near-miss action chunks.
- **VGAS** — keeps the SFT policy fixed and performs inference-time **Best-of-N**
  selection, executing the highest-value candidate chunk.
- **VGAS+** — a post-training **policy extraction** step: candidate chunks are
  optimized in action space under the critic, then distilled back into the VLA through its native
  supervised generative objective, so the refined policy generates high-value chunks directly
  (no inference-time reranking).

Experiments are on the **LIBERO** and **MetaWorld** benchmarks with a **SmolVLA** policy.

---

## Installation

```bash
# 1) Python environment
conda create -n VGAS python=3.10 -y && conda activate VGAS
pip install -r requirements.txt

# 2) LIBERO benchmark (from source)
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO && pip install -e . && cd ..

# 3) MetaWorld / MuJoCo are pulled in by requirements.txt (mujoco, robosuite).
#    For headless rendering the scripts default to EGL (MUJOCO_GL=egl).
```

Key dependencies: `torch 2.7.1` (CUDA 12), `transformers==4.57.0`, `peft==0.17.1`,
`lerobot==0.4.0`, `robosuite==1.4.1`, `mujoco==3.3.6`.

The shell scripts assume the conda env is named `VGAS` and that `conda` / `lerobot` are on `PATH`.
Set `CONDA_BASE`, `LIBERO_ROOT`, `CUDA_VISIBLE_DEVICES`, etc. via environment variables to match
your machine — every script reads them with sensible defaults.

---

## Pretrained checkpoints & datasets

We release the SFT base policy, all four VGAS critics, all four VGAS+ policies, and the
5-shot LIBERO split on Hugging Face:

- Checkpoints: [`SemyonXu616/VGAS-5-shot`](https://huggingface.co/SemyonXu616/VGAS-5-shot)
- 5-shot LIBERO dataset: [`SemyonXu616/HF_LIBERO_5_SHOT`](https://huggingface.co/datasets/SemyonXu616/HF_LIBERO_5_SHOT)

Checkpoint layout:

```text
smolvla/5_SHOT/pretrained_model/       # Shared 5-shot SmolVLA policy
{suite}/vgas_critic/last.ckpt          # VGAS inference-time critic
{suite}/vgas_plus/pretrained_model/    # Distilled VGAS+ policy
```

`{suite}` is one of `goal`, `object`, `spatial`, or `long` (`long` corresponds to
`libero_10`). The released VGAS+ policies are available at:

| LIBERO suite | VGAS+ checkpoint |
|---|---|
| Goal | [`goal/vgas_plus/pretrained_model`](https://huggingface.co/SemyonXu616/VGAS-5-shot/tree/main/goal/vgas_plus/pretrained_model) |
| Object | [`object/vgas_plus/pretrained_model`](https://huggingface.co/SemyonXu616/VGAS-5-shot/tree/main/object/vgas_plus/pretrained_model) |
| Spatial | [`spatial/vgas_plus/pretrained_model`](https://huggingface.co/SemyonXu616/VGAS-5-shot/tree/main/spatial/vgas_plus/pretrained_model) |
| Long (`libero_10`) | [`long/vgas_plus/pretrained_model`](https://huggingface.co/SemyonXu616/VGAS-5-shot/tree/main/long/vgas_plus/pretrained_model) |

Download the checkpoints and the prepared LIBERO split directly into the paths expected by the
provided launchers:

```bash
hf download SemyonXu616/VGAS-5-shot \
  --local-dir pretrained_vla/Libero
hf download SemyonXu616/HF_LIBERO_5_SHOT \
  --repo-type dataset \
  --local-dir dataset/Libero/HF_LIBERO_5_SHOT
```

VGAS and VGAS+ start from the **fine-tuned SFT policy** — you do not need to re-run base SFT.

---

## Data preparation

Both benchmarks share the same reward-annotated few-shot pipeline (5 demonstrations per task).

**LIBERO** (see `data/libero/README.md` for all options):

```bash
bash data/libero/hflibero_fewshot_dataset_pipeline.sh
```

**MetaWorld** (see `data/metaworld/README.md`):

```bash
bash data/metaworld/metaworld_fewshot_dataset_pipeline.sh
```

---

## Training

All runs start from the released SFT policy. VGAS keeps the SFT policy frozen and trains the
**Q-Chunk-Former critic** with the `TD + EGR` objective; VGAS+ additionally
distills critic-optimized chunks back into the policy.
Both methods use `scripts/run_qchunk_offline.py`: VGAS is the default, while `--distill`
enables the two-stage VGAS+ pipeline (critic-only warmup, then critic training plus policy distillation).
Script defaults reproduce the paper configuration (LIBERO example: `libero_goal`;
MetaWorld example: `very_hard`).

**VGAS critic (LIBERO, used for Best-of-N):**

```bash
bash run_scripts/libero/train_goal.sh
```

**VGAS+ policy extraction (LIBERO)** — pick a suite with `DATASET_NAME`:

```bash
DATASET_NAME=libero_goal bash run_scripts/libero/train_distill.sh
```

**VGAS+ policy extraction (MetaWorld)** — pick a difficulty group with `TASK_SPLIT`:

```bash
TASK_SPLIT=very_hard bash run_scripts/metaworld/train_vgas_distill_metaworld.sh
```

---

## Evaluation

**LIBERO**

```bash
# BC baseline (fixed SFT policy, no critic)
bash run_scripts/libero/test_bc.sh

# VGAS on LIBERO-Goal: base policy + critic, inference-time Best-of-N
MODE=vgas \
POLICY_PATH=pretrained_vla/Libero/smolvla/5_SHOT/pretrained_model \
CRITIC_PATH=pretrained_vla/Libero/goal/vgas_critic/last.ckpt \
bash run_scripts/libero/test_vgas.sh

# VGAS+ on LIBERO-Goal: distilled policy, no critic or reranking
MODE=vgas_plus \
POLICY_PATH=pretrained_vla/Libero/goal/vgas_plus/pretrained_model \
bash run_scripts/libero/test_vgas.sh
```

Set `ENV_TASK` and the checkpoint suite together for other evaluations: `libero_object` ↔ `object`,
`libero_spatial` ↔ `spatial`, and `libero_10` ↔ `long`.

**MetaWorld**

```bash
# BC baseline
bash run_scripts/metaworld/eval_metaworld_bc.sh

# VGAS+: the distilled policy (point CKPT_ROOT_REL at your VGAS+ checkpoints)
bash run_scripts/metaworld/eval_metaworld_vgas_plus.sh
```

`run_scripts/metaworld/summarize_success_rate.py` aggregates per-group success rates from the
evaluation outputs.

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{xu2026vgas,
  title   = {VGAS: Value-Guided Action-Chunk Selection for Few-Shot Vision-Language-Action Adaptation},
  author  = {Xu, Changhua and Yu, En and Xuan, Junyu and Lu, Jie},
  journal = {arXiv preprint arXiv:2602.07399},
  year    = {2026}
}

% VGAS+ — preprint coming soon.
% The citation entry will be added here once it is online.
```

## Acknowledgments

This project builds on [SmolVLA](https://github.com/huggingface/lerobot) and the
[LeRobot](https://github.com/huggingface/lerobot) framework, and evaluates on the
[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) and
[MetaWorld](https://github.com/Farama-Foundation/Metaworld) benchmarks. We thank the authors of
these projects for open-sourcing their work.
