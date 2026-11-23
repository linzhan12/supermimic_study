# SuperMimic

[Project page](https://supermimic.github.io) | [Paper](https://arxiv.org/abs/2610.28480)

SuperMimic trains a humanoid whole-body motion tracker with one differentiable world model, built on [mjlab](https://github.com/mujocolab/mjlab).

## Setup

Install [uv](https://docs.astral.sh/uv/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone both repos as siblings:

```bash
mkdir supermimic_ws && cd supermimic_ws
git clone git@github.com:HybridRobotics/supermimic.git
git clone git@github.com:HybridRobotics/mjlab_research.git
cd supermimic
```

## Usage

Available task IDs:

- `SuperMimic-GenCoord-G1` — generalized coordinates representation
- `SuperMimic-MaxCoord-G1` — maximal coordinates representation

Train a world model and policy:

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> uv run src/supermimic/scripts/train.py \
  <TASK_ID> --runner.run_name <description>
```

Play back a trained model:

```bash
uv run src/supermimic/scripts/play.py \
  <TASK_ID> --wandb-path <RUN_PATH>
```

### P0 paired replay on flat ground

Run the existing G1 actor headlessly from the same reference frame in each of
three modes. A uses the full motion target, B uses the reference-derived speed,
and C uses a fixed joystick sequence. The two initial states exercise near-rest
start and release from walking. This experiment runs inference in MuJoCo.

From this directory, with a local checkpoint and the converted LAFAN1 clip:

```bash
P0_OUT=outputs/p0-g1-replay
OMP_NUM_THREADS=2 .venv/bin/python scripts/eval_p0.py \
  --checkpoint-dir logs/SuperMimic-GenCoord-G1/2026-11-17_17-55-41 \
  --motion-file ../data/lafan1_official/walk1_subject1.npz \
  --output-dir "$P0_OUT" && git -C .. rev-parse HEAD > "$P0_OUT/git_commit.txt"
```

Use a new output directory for each run. The example uses frames 0 and 205 of
this clip. Use `--stand-frame` and `--walk-frame` when choosing a different
clip. The evaluator uses CPU by default and reads `local_config.json` when
available. Older checkpoints use the current runner defaults and must match
that architecture. Choose a checkpoint with velocity goal statistics.

`summary.json` records the input hashes, seed, control period, initial states,
completion and per-phase speed error. Completion means the policy reached the
end of the planned window without triggering the shared pelvis height or tilt
limits. Each JSONL trace pairs the actor goal and action at time `t` with the
state and foot contact at `t+1`. The velocity vector is forward and left in
m/s in the pelvis heading frame and world-up yaw in rad/s. For A and B, the
speed error uses the same smoothed, clipped reference speed. The actor's full
reference goal is recorded separately in A. The legacy motion reward is
retained as a diagnostic field.

## Acknowledgements

SuperMimic contains code inspired by these projects:

- [ControlVAE](https://github.com/heyuanYao-pku/Control-VAE) (BSD-3-Clause)
- [robotic_world_model](https://github.com/leggedrobotics/robotic_world_model) (Apache-2.0)
- [DreamerV3](https://github.com/danijar/dreamerv3) (MIT)

## License

Copyright 2026 The SuperMimic Authors.

Released under the [Apache License 2.0](LICENSE).
