# Stage 1 MPI Overview

This directory mirrors the `stage0_serial` GPT‑2 implementation but extends it with MPI-based scale-out. All files with the same names share the original logic plus distributed hooks; entirely new behavior is called out below.

## Core Infrastructure
- `tensor.h` – identical to serial: lightweight tensors with autograd metadata. Serves as the base for both data- and model-parallel code.
- `autograd.h`, `add.h`, `transpose.h` – same functionality as stage0; operations register backward callbacks so gradients flow through MPI collectives added elsewhere.

## Distributed Math & Layers
- `matmul.h`, `softmax.h`, `gelu.h`, `layernorm.h` – now identical to their serial counterparts. The SPMD MPI run keeps these ops fully replicated, letting us focus on data-parallel training without per-op collectives. Each header still exposes a `*_set_distributed` stub so higher-level code can call it without conditionals.

## Model Components
- `multihead_attention.h` – matches the serial implementation; all ranks run full attention locally. The `mha_set_distributed` stub remains for API compatibility but is a no-op after simplifying the column sharding logic.
- `head.h`, `mlp.h`, `linear.h`, `block.h`, `gpt.h` – unchanged math but now call the distributed helpers (`mha_set_distributed`, `layernorm_set_distributed`, `matmul_set_distributed`, etc.) during initialization.
- `embedding.h` – same as serial; embeddings stay fully replicated so every rank can index tokens/positions independently.

## Training Utilities
- `tokenizer.h` – adds MPI broadcast helpers (`tokenizer_broadcast`) so rank 0 can build the vocab and share serialized state with other ranks.
- `dataloader.h` – supports distributed sampling: rank 0 prepares tokens, broadcasts them, and `dataloader_next_batch` offsets each rank into a distinct slice so the global batch equals `batch_size * world_size`.
- `cross_entropy.h`, `checkpoint.h`, `broadcast.h` – identical logic but used in MPI-aware entrypoints.
- `adam.h` – optimizer now tracks rank/world size; after each `adam_step` it broadcasts updated parameters/moving averages so replicas stay in sync.

## Executables
- `train_gpt2.cpp` – MPI entrypoint. Upgrades relative to stage0:
  - Global batch is divided among ranks (“single program, multiple data”) to keep total tokens constant.
  - Training loop collects per-rank loss via `MPI_Allreduce`.
  - Gradients are flattened into a contiguous buffer for a single `MPI_Allreduce` per step, then unpacked before Adam updates.
  - Text generation remains identical to the serial path but runs only on rank 0 after training for easy timing comparisons.
  - Logs include token counts, world size, and timing to highlight distributed throughput.
- `inference.cpp` – MPI-aware checkpoint loader. Rank 0 loads weights, broadcasts them, and every rank runs the same inference logic; only rank 0 prints results.
- `train_gpt2` / `inference` binaries – compiled MPI executables; rebuild with `mpicxx`.

## Data Files
- `dummy_data.txt`, `dummy_data2.txt`, `trained_weights.bin` – identical content as stage0 but stored here for convenience when running MPI jobs in this directory.

In short, `stage1_mpi` preserves the clean C implementation from `stage0_serial` while layering in MPI primitives for **data parallel** scale-out: rank 0 prepares data, all ranks own full model replicas, gradients are averaged every step, and rank 0 handles logging/generation. Use `mpirun -np <N> ./train_gpt2` to exercise these upgrades.

## Code Flow (train_gpt2.cpp)
- Parse optional `--seed`, init MPI, and seed RNG.
- Validate `global_batch_size` divisibility by `world_size`; derive per-rank `batch_size`.
- Rank 0 tokenizes/pads corpus, then broadcasts tokenizer state to all ranks.
- Initialize GPT, collect params, build sync metadata (shared vs local), and broadcast initial shared parameters from rank 0.
- Set up Adam with distributed masks; build dataloader, broadcast token buffer, and offset batches per rank.
- Training loop: get per-rank batch → forward → allreduce loss for logging → backward → pack grads → allreduce grad buffer → unpack/scale → Adam step → zero grads → rank 0 logs loss.
- After training: barrier, rank 0 saves weights, then distributed greedy generation (rank 0 selects next token, broadcasts). Timings collected via barriers/reduces. Cleanup and `MPI_Finalize`.
