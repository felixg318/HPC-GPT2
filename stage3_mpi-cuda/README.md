# Stage 3 MPI + CUDA Overview

Hybrid MPI data-parallel training on GPUs. Each rank holds a full model replica; CUDA kernels handle compute, MPI synchronizes parameters/gradients.

## Headers
- Core tensor/autograd: `tensor.h`, `autograd.h`
- CUDA math: `matmul.h`, `softmax.h`, `gelu.h`, `layernorm.h`, `add.h`, `transpose.h`, `cuda_kernels.cu`
- Model: `embedding.h`, `linear.h`, `head.h`, `multihead_attention.h`, `mlp.h`, `block.h`, `gpt.h`
- MPI utilities: `mpi_utils.h` (rank/world helpers), `broadcast.h`
- Training/runtime: `tokenizer.h`, `dataloader.h` (rank-aware), `cross_entropy.h`, `adam.h`, `checkpoint.h`

## Code Flow (train_gpt2.cpp)
- MPI init; derive `rank`/`world_size`; choose CUDA device `rank % device_count`.
- Rank 0 tokenizes/pads `dummy_data.txt`, then broadcasts tokenizer state. Vocab size drives model init.
- Hyperparameters: block_size 48, n_layer 8, n_head 12, n_embd 192, lr 3e-3, global_batch_size 16 (must divide world_size), epochs 8.
- Compute per-rank `batch_size = global_batch_size / world_size`; init GPT (with distributed flag), collect params, set up Adam (linear LR decay); broadcast weights for consistency.
- Dataloader seeded with `rank`/`world_size` to split data; training loop per rank: batch → forward → backward → `mpi_allreduce_grads` → Adam step → zero grad → rank 0 logs allreduced loss.
- After training: rank 0 saves checkpoint; rank 0 runs greedy text generation; print timings; free resources and `MPI_Finalize`.

## Notable Differences / Issues
- Still **data parallel**, not model parallel: every rank holds full weights; generation runs only on rank 0 (other ranks idle).
- Learning rate `3e-3`, unlike stages 0/1 (`3e-4`); adjust if you want parity.
- Requires `global_batch_size` divisible by `world_size`; otherwise it exits.
