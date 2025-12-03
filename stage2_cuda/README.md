# Stage 2 CUDA Overview

CUDA-enabled single-process training of the GPT-2 mini model. Mirrors the header graph from stage0 but uses CUDA kernels under the hood and optional MPI helpers for gradient allreduce (data parallel).

## Headers
- Core tensor/autograd: `tensor.h`, `autograd.h`
- Math/kernels: `matmul.h`, `softmax.h`, `gelu.h`, `layernorm.h`, `add.h`, `transpose.h`, `cuda_kernels.cu`
- Model pieces: `embedding.h`, `linear.h`, `head.h`, `multihead_attention.h`, `mlp.h`, `block.h`, `gpt.h`
- Training utils: `tokenizer.h`, `dataloader.h`, `cross_entropy.h`, `adam.h`

## Code Flow (train_gpt2.cpp)
- Optional MPI init for rank/world size; pick CUDA device by rank modulo device count.
- Tokenize/pad `dummy_data.txt`, compute vocab size.
- Hyperparameters (block_size 48, n_layer 8, n_head 12, n_embd 192, lr 3e-3, batch_size 4, epochs 8).
- Init GPT, collect parameters, set up Adam (linear LR decay). If MPI with multiple ranks, broadcast initial weights.
- Init dataloader over token stream; log expected token counts on master rank.
- Training loop: dataloader batch → `gpt_forward_with_loss` → `backward` → optional MPI gradient allreduce → `adam_step` → `adam_zero_grad` → master logs loss.
- After training: log timing, run greedy text generation on master, print timing, free resources (GPT/optimizer/dataloader/tokenizer/param list), and `MPI_Finalize` if enabled.

## Notable Differences / Issues
- Learning rate is `3e-3`, higher than stages 0/1 (`3e-4`). If parity is desired, reduce to match.
- When compiled with MPI and run on multiple ranks, `batch_size` is **not** divided by `world_size`; all ranks consume identical batches, scaling the effective global batch and duplicating data. For proper data parallelism, the batch should be split or the dataloader should offset per rank.
