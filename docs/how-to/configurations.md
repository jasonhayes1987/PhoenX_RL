# Configuration Files

Every training and evaluation run is defined by a YAML file under `configs/`.
The layout is one directory per environment, one file per algorithm:

```
configs/
├── LunarLander-v3/
│   ├── sac.yml
│   └── ppo.yml
└── <EnvironmentName>/
    └── <algorithm>.yml
```

<!-- TODO(docs-writer): verify this tree against the repo and correct it. -->

## Anatomy of a config

<!-- TODO(docs-writer): paste a short real config (e.g. LunarLander sac.yml)
     and annotate every top-level key: agent/algorithm selection, environment
     spec, network architecture, training hyperparameters, buffer/HER options,
     logging/W&B, save paths. This section is the single most useful page in
     the docs — be thorough. -->

## Key reference

<!-- TODO(docs-writer): table of all recognized config keys per section, with
     type, default, and effect. Source of truth: phoenx.builder (load_config /
     build_trainer_from_config) — document what the code actually reads, not
     what seems plausible. -->

## Creating a config for a new environment

<!-- TODO(docs-writer): minimal steps — copy nearest existing config, keys that
     must change per environment, common pitfalls (obs/action space mismatches). -->
