# VGAS / VGAS+ Training Algorithm Flow

This document describes the actual algorithm implemented in:

- `scripts/train_qchunk_offline.py`
- `qchunk/distill_helpers.py`

## 1) Components

- `policy`: the pretrained SmolVLA; frozen for VGAS and trainable during VGAS+ policy updates.
- `teacher_policy`: optional frozen SmolVLA, created only when a VGAS+ phase requests teacher actions.
- `critic`: Q-chunk critic, trained throughout every VGAS and VGAS+ run.

The shared trainer always updates the critic. With `distill.enable=False` (the CLI default),
the policy stays frozen and the run is VGAS. Passing `--distill` sets `distill.enable=True`
and activates the two-stage VGAS+ policy-distillation path below.

## 2) Phase Split

The phase split applies only to VGAS+ (`distill.enable=True`):

Per training step `step`:

- Phase-1 if `step < distill.phase1_steps`
- Phase-2 if `step >= distill.phase1_steps`

VGAS (`distill.enable=False`) has no phase switch and trains only the critic for all configured steps.
The VGAS+ phase switch is based on global step, so resume continues in the correct phase.

## 3) Per-Step Training Order

For each batch:

1. Build/prepare batch (and next observations for critic TD target).
2. Update critic (`vgas_policy.update_critic`).
3. For VGAS+ only, update the student policy according to the active phase. The paper configuration
   uses a critic-only phase-1 warmup (`distill.phase1_train_student=False`).

The critic is updated in every phase; the policy is updated only when VGAS+ enables it.

## 4) Critic Update Logic

### VGAS

- OOD source and TD target policy: frozen pretrained `policy`.
- The critic is updated for all configured steps; the policy is never updated.

### VGAS+ Phase-1 (critic warmup)

- If `distill.phase1_use_student_as_source=True`, both OOD source and TD target use `policy`.
- Otherwise, both use `teacher_policy`.
- The paper configuration uses the student as the source and keeps policy updates disabled during warmup.

### VGAS+ Phase-2

- OOD source for critic: `student_policy`
- TD target policy: `student_policy`
- If `distill.phase2_disable_critic_best_of_n=True`:
  - force `target_action_samples=1` (disable best-of-n for critic target).

## 5) Actor (Student) Update Logic

### Optional Phase-1 Distillation

When `distill.phase1_train_student=False`, this entire policy update is skipped and phase-1 is a
critic-only warmup. The following behavior applies only when that ablation flag is enabled.

Teacher target action chunk is built from:

- if critic exists: sample teacher actions and choose the best by critic Q (`_select_best_teacher_actions`)
- else: direct single teacher action chunk

Then replace batch action target via `_build_distill_actor_batch` and run normal `update_policy(...)`.

### Phase-2 Dual Distillation

Build two targets:

1. **Teacher anchor target**
   - single-sample teacher action (`_sample_policy_actions_single`) -> `teacher_batch`

2. **Optimized student target**
   - from `_select_local_optimized_student_actions(...)` -> `optimized_batch`

Then optimize student with:

`L_total = lambda_w2 * L_w2 + lambda_opt * L_opt`

- `L_w2`: student forward loss on `teacher_batch`
- `L_opt`: student forward loss on `optimized_batch`
- both share the same sampled `noise/time` per step (`update_policy_with_dual_distill`)

## 6) Phase-2 Global + Local Optimization Details

Implemented in `qchunk/distill_helpers.py::_select_local_optimized_student_actions`.

### Global step

1. Student samples `K = phase2_global_samples` candidate action chunks.
2. Optional ablation: if `phase2_include_dataset_action_in_global_pool=True`, append dataset action to global pool.
3. Critic scores all candidates.
4. Keep top-`m = phase2_local_samples`.

### Local step

For each local iteration (`phase2_local_opt_steps`):

1. Compute gradient ascent on critic Q wrt action chunk.
2. Optional grad normalization (`phase2_local_grad_normalize`).
3. Optional advantage-style weighting (`phase2_local_adv_weight`, epsilon `phase2_local_adv_eps`).
4. Action clip to dataset-stat bounds (fallback `[-1, 1]` in normalized space).
5. Keep best-so-far local action per candidate across iterations.

After local iterations, pick argmax-Q action and use it as final optimized target for distillation.

## 7) Important Distill Hyperparameters

- `distill.phase1_steps`
- `distill.phase2_global_samples`
- `distill.local_samples` (phase-2 top-m)
- `distill.phase2_local_opt_steps`
- `distill.phase2_local_opt_lr`
- `distill.phase2_local_grad_normalize`
- `distill.phase2_local_adv_weight`
- `distill.phase2_local_adv_eps`
- `distill.phase2_include_dataset_action_in_global_pool`
- `distill.mask_padded_action_loss` (student-only; mask padded action dims from policy loss)
- `distill.lambda_w2`, `distill.lambda_opt`
- `distill.phase2_disable_critic_best_of_n`
- `distill.student_num_steps`, `distill.teacher_num_steps` (teacher usually fixed/frozen)
