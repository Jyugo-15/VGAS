# Data Utilities

This directory contains the HFLibero/LeRobot dataset tooling. The **primary entry
point** is `data/hflibero_fewshot_dataset_pipeline.sh`, which produces a 5-shot
merged dataset, annotates rewards, then splits into per-suit datasets.

---
We provide the 5-shot dataset used in our experiments on Hugging Face:
https://huggingface.co/datasets/SemyonXu616/HF_LIBERO_5_SHOT

## `hflibero_fewshot_dataset_pipeline.sh`

This script implements the **one-pass** workflow we want:

1. Build a merged few-shot dataset from the original mixed dataset.
2. Annotate rewards on the merged dataset.
3. Split the merged dataset into per-suit datasets.

This guarantees the per-suit datasets are **exactly** the samples included in
`merge` (no re-sampling).

**Default output layout**
```
<OUT_ROOT>/
  merge/
  libero_object/
  libero_goal/
  libero_spatial/
  libero_10/
```

**Key environment variables**
- `DATASET_ROOT`: merged HFLibero root (default: `dataset/Libero/HFlibero`). The raw dataset is downloaded from
  `https://huggingface.co/datasets/HuggingFaceVLA/libero`.
- `OUT_ROOT`: output root (default: `dataset/Libero/HF_LIBERO_5_SHOT_TEST`)
- `MAPPING_JSON`: suite mapping (default: `data/libero_tasksuit_info/libero_task_suites.json`)
- `FEW_SHOT_PER_TASK`: episodes per task (default: `5`)
- `FEW_SHOT_MODE`: `random` or `sequential` (default: `random`)
- `FEW_SHOT_SEED`: RNG seed for random sampling (default: `0`)
- `SUITES`: suit list (default: `libero_object libero_goal libero_spatial libero_10`)
- `REWARD_N_LAST`, `REWARD_VALUE`, `DEFAULT_REWARD`, `REWARD_DISCOUNT`
- `FORCE=true|false`: overwrite outputs (default: `true`)

**Example**
```bash
DATASET_ROOT=dataset/Libero/HFlibero \
OUT_ROOT=dataset/Libero/HF_LIBERO_5_SHOT \
FEW_SHOT_PER_TASK=5 \
FEW_SHOT_MODE=random \
FEW_SHOT_SEED=0 \
bash data/hflibero_fewshot_dataset_pipeline.sh
```

---

## Supporting scripts

### `annotate_rewards.py`
Adds a scalar reward column to each frame. It marks the last `--n-last` steps
with `--reward-value` and fills the preceding frames with `--default-reward`.

```bash
python data/annotate_rewards.py \
  --dataset-root /path/to/libero_goal \
  --output-root /path/to/libero_goal_with_rewards \
  --n-last 3 --reward-value 1.0 --default-reward 0.0 --overwrite
```

### `split_hflibero_by_suite.py`
Splits a merged dataset into per-suit directories (no few-shot).

```bash
python data/split_hflibero_by_suite.py \
  --dataset-root /path/to/merge \
  --output-root /path/to/split \
  --suites libero_object libero_goal
```

### `split_hflibero_few_shot.py`
Creates a single merged few-shot dataset (no per-suit split).

```bash
python data/split_hflibero_few_shot.py \
  --dataset-root /path/to/HFlibero \
  --output-root /path/to/merge \
  --few-shot-per-task 5 --few-shot-mode random --few-shot-seed 0
```

