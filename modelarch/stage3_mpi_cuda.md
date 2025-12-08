# Stage3 MPI+CUDA GPT-2 Architecture

Reference: `stage3_mpi-cuda/train_gpt2.cpp` and headers (`gpt.h`, `block.h`, `multihead_attention.h`, `head.h`, `mlp.h`, `layernorm.h`, `linear.h`, `softmax.h`, `matmul.h`, `dataloader.h`, `tokenizer.h`, `adam.h`, `autograd.h`, `mpi_utils.h`, CUDA helpers).

## Unique Traits
- Combined MPI data parallelism and CUDA kernels: each rank owns a GPU (`rank % device_count`) and a shard of the global batch; grads summed via `mpi_allreduce_grads`.
- Rank-aware dataloader shards token stream; tokenizer is read on rank 0 then broadcast to keep vocab/encoded data identical.
- Distributed helpers (`mpi_utils.h`) handle parameter broadcast (`mpi_broadcast_parameters`) and loss reduction (`mpi_allreduce_loss`).
- `gpt_set_distributed` configures ops for data parallel settings while keeping compute on GPUs.

## Data Flow
- MPI init (`MPI_Init`); RNG seed via `tensor_set_seed`; CUDA device chosen per rank (`rank % device_count`).
- Rank 0 tokenizes `../data/dummy.txt`: `tokenizer_extract` → `tokenizer_encode`; broadcast success flag; `tokenizer_broadcast` shares vocab/encoded; `tokenizer_pad_to` uses `global_batch_size`.
- Per-rank batch size = `global_batch_size / world_size`. `DataLoader.init_with_tokenizer` stores rank/world_size for sharded sampling.
- Model init `gpt_init` + `gpt_set_distributed` (configures distributed matmul/softmax/gelu/xent); collect params; `adam_init` (LR scheduler). If multi-rank, rank 0 broadcasts parameters (`mpi_broadcast_parameters`) to sync initial weights.
- Training loop:
  - `dataloader_next_batch` slices token stream with rank offset.
  - Forward: `gpt_forward_with_loss` → `logits`, `loss`.
  - Backward: `backward(loss)`; clear activations.
  - `mpi_allreduce_grads` sums grads across ranks (data parallel).
  - Optimizer: `adam_step` (with optional grad clipping) → `adam_zero_grad`.
  - Loss averaged via `mpi_allreduce_loss`; master rank logs epoch loss.
- Master saves weights (`save_weights("trained_weights.bin")`).
- Generation (master): sliding window `gpt_forward_logits`, greedy decode until EOS/limit; prints decoded tokens.

## Model Stack (GPT-2 topology on MPI+CUDA runtime)
- Token IDs `(B,T)` → token embedding `wte`; position embedding `wpe`; `broadcast_add` → `x0 (B,T,n_embd)`.
- Repeat `n_layer` blocks:
  - `LN1(x)` → Multi-Head Attention (per head `q/k/v` linears; scaled dot-product with causal mask; softmax; value weighted sum; concat heads; projection).
  - Residual: `x + attn_out` → `x1`.
  - `LN2(x1)` → `MLP` (`c_fc: n_embd→4*n_embd` → `GELU` → `c_proj: 4*n_embd→n_embd`).
  - Residual: `x1 + mlp_out` → next `x`.
- Final `LN_f` → `lm_head Linear: n_embd→vocab_size` → `logits (B,T,V)` → CE loss during training.
