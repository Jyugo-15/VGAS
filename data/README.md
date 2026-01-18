# Data Utilities

This directory collects helper scripts around the HFLibero / LeRobot dataset
pipeline. The subsections below summarise what each entry does, its most
important CLI arguments, and a sample invocation.

---

## `annotate_rewards.py`

Adds a scalar reward column to every frame of a LeRobot dataset. It marks the
last `--n-last` steps in each episode with `--reward-value` and fills the
preceding frames with `--default-reward`.

- **Key args**
  - `--dataset-root`: path to an existing LeRobot dataset (either the merged
    HFLibero root or a per-suit split).
  - `--output-root`: optional destination; when provided the dataset is copied
    here before rewards are written.
  - `--n-last`, `--reward-value`, `--default-reward`, `--overwrite`.
- **Example**
  ```bash
  python data/annotate_rewards.py \
    --dataset-root /data/HFlibero_split/libero_object \
    --output-root /data/HFlibero_split/libero_object_with_rewards \
    --n-last 5 --reward-value 1.0 --default-reward 0.0 --overwrite
  ```

---

## `lerobot_reward_dataset.py`

Defines `RewardAugmentedLeRobotDataset`, a thin subclass of
`lerobot.datasets.LeRobotDataset` that groups frames into episodes and exposes
reward/terminal signals. Running the file directly performs a smoke test or
inspects a dataset.

- **Highlights**
  - Episode-level helpers: `get_episode`, `load_episode`, `get_episode_length`.
  - Chunked sampling with action/reward windows for chunk-based training.
  - CLI preview flags such as `--root`, `--episode`, `--chunk-size`,
    `--batch-size`, etc., mirroring the training datamodule defaults.
- **Example**
  ```bash
  python data/lerobot_reward_dataset.py \
    --root /data/HFlibero_split/libero_object_with_rewards \
    --episode 3 --preview 10 --chunk-size 5 --batch-size 4
  ```

---

## `push_to_hub.py`

Publishes one or more per-suit datasets to the Hugging Face Hub using
`LeRobotDataset.push_to_hub`.

- **Key args**
  - `--split-root`: directory containing suit subfolders.
  - `--repo-template`: template such as `user/HFlibero-{suit}`.
  - `--suites`: optional filter (defaults to every child folder).
  - `--branch`, `--tag`, `--skip-clean`, `--delete-patterns`, `--dry-run`.
- **Example**
  ```bash
  python data/push_to_hub.py \
    --split-root /data/HFlibero_split \
    --repo-template your-user/HFlibero-{suit} \
    --suites libero_object libero_goal \
    --branch main
  ```

---

## `push_to_hub_preview.py`

Creates a lightweight `datasets` preview with image columns so Hugging Face
renders frames in the browser. It loads one suit directory, casts the image
columns to `datasets.Image`, and uploads the result.

- **Key args**
  - `--split-root`, `--suit`, `--repo-id`, `--branch`, `--max-frames`.
- **Example**
  ```bash
  python data/push_to_hub_preview.py \
    --split-root /data/HFlibero_split \
    --suit libero_object \
    --repo-id your-user/HFlibero-object-preview \
    --max-frames 5000
  ```

---

## Split scripts overview

Three closely related scripts reshape the merged HFLibero dataset. They share
many arguments (`--dataset-root`, `--mapping-json`, `--output-root`, `--suites`,
`--force`, `--few-shot-*`). The table below highlights how their outputs differ.

| Script | Output layout | Few-shot behaviour |
| ------ | ------------- | ------------------ |
| `split_hflibero_by_suite.py` | Creates **one directory per suit** (e.g. `libero_object`, `libero_goal`). All episodes for the selected suits are kept. | No few-shot support. |
| `split_hflibero_by_suite_few_shot.py` | Same per-suit directories as above, but each task inside a suit keeps at most `--few-shot-per-task` episodes (deterministic or random). | Uses `--few-shot-per-task`, `--few-shot-mode`, `--few-shot-seed`. |
| `split_hflibero_few_shot.py` | Produces a **single merged dataset** at `--output-root` that mirrors the original layout but drops episodes beyond the few-shot cap (optionally limited to certain suits). | Same few-shot flags; no per-suit splitting. |

### Shared parameters

- `--dataset-root`: merged HFLibero dataset (default `/.../dataset/HFlibero`).
- `--mapping-json`: mapping from LIBERO suits to task descriptions.
- `--suites`: optional subset of suits.
- `--output-root`: destination directory (per-suit scripts create subfolders,
  single few-shot writes directly here).
- `--force`: remove existing outputs.

### Few-shot specific parameters

- `--few-shot-per-task`: maximum number of episodes per task (optional).
- `--few-shot-mode`: `sequential` (take earliest episodes) or `random`.
- `--few-shot-seed`: RNG seed when using random sampling.

### Example commands

```bash
# Split merged dataset into per-suit folders
python data/split_hflibero_by_suite.py \
  --dataset-root /data/HFlibero \
  --output-root /data/HFlibero_split \
  --suites libero_object libero_goal

# Same split with few-shot cap (per suit)
python data/split_hflibero_by_suite_few_shot.py \
  --dataset-root /data/HFlibero \
  --output-root /data/HFlibero_split_fewshot \
  --few-shot-per-task 10 --few-shot-mode random --few-shot-seed 42

# Keep a single merged dataset but truncate every task to 5 episodes
python data/split_hflibero_few_shot.py \
  --dataset-root /data/HFlibero \
  --output-root /data/HFlibero_fewshot \
  --few-shot-per-task 5 --few-shot-mode sequential
```

Use the variant that matches your downstream workflow: per-suit splits when you
need isolated datasets for each LIBERO suite, the per-suit few-shot wrapper when
you still want one directory per suit but fewer episodes, or the merged few-shot
clone when you prefer a single dataset that already aggregates all suits.

---

## `scripts/hflibero_dataset_pipeline.sh`

Wrapper script that ties the stages above together:

1. Optionally split the merged dataset (`split_hflibero_by_suite*.py`/`split_hflibero_few_shot.py`).
2. Optionally annotate rewards (`annotate_rewards.py`).
3. Optionally push datasets and/or preview datasets to the Hugging Face Hub.

The script now accepts command-line flags (defaults mirror the env vars defined
inside the file). Run `./scripts/hflibero_dataset_pipeline.sh --help` for the
full list:

- `--dataset-root`, `--mapping-json`
- `--split-mode {none,per_suite,per_suite_few_shot,merged_few_shot}`
- `--suites "libero_object ..."`
- `--few-shot-per-task`, `--few-shot-mode`, `--few-shot-seed`
- `--add-reward / --no-add-reward`, `--reward-n-last`, `--reward-value`, `--default-reward`
- `--push-to-hub / --no-push-to-hub`, `--repo-template`, `--hf-username`, `--hf-branch`, `--hf-private`
- `--push-preview / --no-push-preview`, `--preview-repo-template`, `--preview-max-frames`
- plus additional paths for intermediate outputs (`--split-root`, `--reward-root`, etc.)

### Example

```bash
./scripts/hflibero_dataset_pipeline.sh \
  --split-mode per_suite_few_shot \
  --suites "libero_object libero_goal" \
  --few-shot-per-task 5 \
  --add-reward \
  --reward-n-last 3 \
  --push-to-hub \
  --hf-username myuser \
  --hf-private \
  --push-preview
```

Combine the flags to match your workflow; any omitted option falls back to the
defaults embedded at the top of the script.
- Push/preview uploads are opt-in: pass `--push-to-hub` and/or `--push-preview` to enable those stages (defaults keep everything local).
