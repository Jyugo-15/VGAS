# VGAS (Value-Guided Action-Chunk Selection)

This repository contains the code for **VGAS**, a generation--selection framework for few-shot Vision-Language-Action (VLA) adaptation. VGAS combines a high-recall VLA policy for action-chunk proposal with a geometrically grounded Transformer critic (**Q-Chunk-Former**) and **Explicit Geometric Regularization (EGR)** to improve ranking resolution under limited supervision.

## Method ↔ Code Map

- **VGAS (generation--selection / Best-of-N)**: `scripts/eval_qc_bestofn.py`, `smolvla_qchunk/eval/bestofn_eval.py`, `qchunk/vgas_policy.py`
- **Q-Chunk-Former critic**: `qchunk/valuequeryhead.py`, `qchunk/qchunked_critic.py`
- **EGR (explicit geometric regularization)**: `qchunk/ood_calql_utils.py`, `qchunk/qchunked_critic.py`
- **Training pipeline**: `scripts/run_qchunk_offline.py`, `scripts/train_qchunk_offline.py`
- **Data utilities**: `data/README.md`

## Environment

- Python and CUDA setup are assumed.
- Install dependencies:

```bash
pip install -r requirements.txt

git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e .
```

The provided shell scripts assume a conda environment named `VGAS`. Adjust the scripts if your environment differs.

## Data

The pipeline targets LIBERO/LeRobot-style datasets. See `data/README.md` for dataset preparation and reward annotation utilities. Update dataset paths in the shell scripts and/or command-line arguments to match your local setup.

## Training (Critic / Q-Chunk-Former)

Example entry point:

```bash
python scripts/run_qchunk_offline.py \
  --policy-path <PRETRAINED_POLICY_DIR> \
  --dataset-root <DATASET_ROOT> \
  --dataset-repo-id libero_goal \
  --chunk-size 32 \
  --q-chunk-len 32 \
  --critic-type q_chunk_former \
  --use-ood-reg true \
  --loss-rank-weight 5.0
```

For a full configuration example, see `run_scrpit/train_goal.sh`.

## Evaluation (BC vs Best-of-N)

BC (no critic guidance):

```bash
python scripts/eval_qc_bestofn.py \
  --env-task libero_goal \
  --policy-path <PRETRAINED_POLICY_DIR> \
  --no-use-best-of-n \
  --best-of-n 1
```

Best-of-N (critic-guided):

```bash
python scripts/eval_qc_bestofn.py \
  --env-task libero_goal \
  --policy-path <PRETRAINED_POLICY_DIR> \
  --critic-state <CRITIC_CKPT> \
  --use-best-of-n \
  --best-of-n 8
```

## Reproducing Experiments

Our experiment results can be reproduced by running `run_scrpit/test_bc.sh` (BC) and `run_scrpit/test_vgas.sh` (VGAS). (Libero goal as an example)

The fine-tuned 5-shot SmolVLA and trained critic checkpoints are available at:
https://huggingface.co/SemyonXu616/VGAS-5-shot/tree/main

## TODO

- Add additional benchmarks (e.g., MetaWorld).
- Support alternative policy classes (e.g., diffusion-based policies).

## Notes


- Adjust paths, GPU IDs, and seeds in the scripts to fit your local environment.
