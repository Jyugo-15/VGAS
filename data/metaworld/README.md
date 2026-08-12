# MetaWorld Data Utilities

This directory contains a MetaWorld dataset pipeline aligned with the LIBERO scripts under `data/`.

## Scripts

- `split_metaworld_few_shot.py`
  - Builds a merged few-shot dataset by sampling episodes **per `task_id`**.
- `annotate_metaworld_rewards.py`
  - Adds your custom `reward` labels (no `mc_returns`).
  - Optional `terminal` generation modes: `none`, `episode_end`, `next_success`.
- `split_metaworld_by_group.py`
  - Splits a merged dataset into `easy/medium/hard/very_hard` subsets.
- `metaworld_fewshot_dataset_pipeline.sh`
  - Orchestrates few-shot merge + reward annotation, and optional group split.
- `metaworld_task_groups.json`
  - Repo-local MetaWorld task group mapping.

## Quick Start

```bash
DATASET_ROOT=dataset/MetaWorld/MT50_50_SHOT \
OUT_ROOT=dataset/MetaWorld/HF_METAWORLD_5_SHOT \
FEW_SHOT_PER_TASK=5 \
FEW_SHOT_MODE=random \
FEW_SHOT_SEED=0 \
TERMINAL_MODE=none \
SPLIT_GROUPS=false \
bash data/metaworld/metaworld_fewshot_dataset_pipeline.sh
```

Enable group split:

```bash
SPLIT_GROUPS=true \
GROUPS="easy medium hard very_hard" \
bash data/metaworld/metaworld_fewshot_dataset_pipeline.sh
```
