# Isaac Sim Environments

PhoenX integrates with [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) for
GPU-accelerated robotics environments.

## Environment setup

Follow NVIDIA's installation guide to create the Isaac Lab environment
(conda, Python version matching your Isaac Sim release), then install PhoenX
into the same environment. See [Getting Started](getting-started.md).

## Verify your installation

After installing PhoenX into your Isaac Lab environment, confirm that Isaac
Sim, CUDA, and GPU environments work:

```bash
pytest tests/test_isaac_setup.py -v
```

These tests are part of the main suite (`pytest tests`) but are gated by the
`isaac` marker and auto-skip when `isaaclab` or CUDA is unavailable. To exclude
them on machines without Isaac:

```bash
pytest -m "not isaac"
```

You can also run the module standalone inside the Isaac container:
`python tests/test_isaac_setup.py`.

## Training on Isaac Lab tasks

<!-- TODO(docs-writer): worked example — a config that targets an Isaac Lab
     task, the phoenx-train invocation, expected startup behavior (first-run
     extension pulls can take ~10 minutes), headless vs. rendered operation. -->

## Custom environment configurations

PhoenX ships custom Isaac Lab environment configs (e.g. the Franka variants in
the repo).

<!-- TODO(docs-writer): document each custom *_cfg.py — what task it defines,
     how it differs from the stock Isaac Lab task, and how a user selects it
     from a YAML config. Explain how to write a new one, referencing Isaac
     Lab's own task-creation docs rather than duplicating them. -->
