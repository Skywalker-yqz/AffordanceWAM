# Data

AffordanceWAM learns shared interaction representations from human videos and robot trajectories, followed by task-specific robot post-training.

| Stage | Sources | Scale |
|-------|---------|-------|
| Human pre-training | Ego4D-FHO, EPIC-KITCHENS VISOR, HOI4D, H2O, MECCANO | More than 300 hours |
| Robot pre-training | RoboInter (DROID and RH20T), BridgeData V2, InternData-A1, RoboCasa365 | Approximately 0.4 million trajectories |
| RoboCasa post-training | 24 tasks | 300 demonstrations per task |
| CALVIN post-training | ABC training split | Complete split |
| Real-world post-training | 8 tasks | 50 trajectories per task |

Stage I samples human and robot data at a 1:1 ratio. Training uses 17-frame clips at 10 FPS and 256×320 resolution, with five observed frames and twelve future frames. Validation uses a fixed set of 2,000 clips and a stratified 1% holdout.

Store local manifests and normalization statistics under `data/preprocessed/` and supply their paths to the training or inference entry point.
