# Generative Circuit Design for Quantum-Selected Configuration Interaction

Training/evaluation pipeline for **GQE (Generative Quantum Eigensolver) + QSCI (Quantum-Selected Configuration Interaction)** on molecular Hamiltonians.  
The entrypoint is `train.py` (Hydra), and quantum-circuit sampling is performed with **CUDA-Q**.

## Environment (guideline)
- Python 3.11
- NVIDIA GPU (when using Docker: `--gpus all`)
- CUDA-Q / CUDA-QX (Docker is the easiest way to get a working setup)

## Installation (local)

This repository’s `Dockerfile` is based on `ghcr.io/nvidia/cudaqx:0.4.0`.

### build

```bash
docker build -t gqe_qsci .
```

### run

```bash
docker run -it --rm \
  --gpus all \
  -v "$(pwd):/workspace" \
  -w /workspace \
  gqe_qsci
```
## Running (basic)

Hydra configs live under `configs/`. You can override them via `group=name` and/or `key=value`:

```bash
python train.py molecule=n2 trainer.epochs=200
```


## Outputs and resuming (checkpoints / W&B)

By default (`configs/default.yaml`), outputs are written to:
- **Output dir**: `outputs/${project.name}/${exp_tag}`
- **Checkpoint**: `.../models/last.ckpt` (loaded if present)
- **Replay buffer**: `.../replay_buffer.pkl`
- **W&B run id**: `.../run_id` (re-running in the same directory resumes with `resume='allow'`)

To resume:
- Run with the same `exp_tag` (i.e., the same `output` directory)
- Keep `trainer.load_checkpoint=true` (default)

Example (pin `exp_tag` for easy resuming):

```bash
python train.py molecule=n2 exp_tag=my-n2-run
```

To start fresh (do not load checkpoints):

```bash
python train.py trainer.load_checkpoint=false
```

## License

Apache License 2.0 (see `LICENSE`)