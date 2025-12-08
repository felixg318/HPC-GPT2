# Stage2 CUDA GPT-2 Architecture

Reference: `stage2_cuda/train_gpt2.cpp` and headers (`gpt.h`, `block.h`, `multihead_attention.h`, `head.h`, `mlp.h`, `layernorm.h`, `linear.h`, `softmax.h`, `matmul.h`, `dataloader.h`, `tokenizer.h`, `adam.h`, `autograd.h`, `checkpoint.h`, CUDA helpers).

## Unique Traits
- CUDA-backed kernels for matmul/softmax/GELU/layernorm etc. (`cuda_kernels.cu`, `cuda_utils.h`) with device selection per rank.
- Optional MPI data parallelism: rank 0 broadcasts initial params; `mpi_allreduce_grads` sums gradients when `world_size>1`.
- Single-process tokenizer/dataloader per rank (no broadcast); data-parallel sync is only through gradients/params.

## Data Flow
- Optional MPI init (`USE_MPI`); CUDA device selection per rank (`cudaGetDeviceCount`, `cudaSetDevice`). RNG seeded via `tensor_set_seed`.
- Each rank tokenizes `../data/dummy.txt`: `tokenizer_extract` → `tokenizer_encode` → `tokenizer_pad_to`; vocab size reused for all ranks.
- Model init `gpt_init`; collect params; `adam_init` (LR scheduler `linear_lr_decay`). If `world_size>1`, rank 0 broadcasts parameters (`mpi_broadcast_parameters`) for identical starts.
- `DataLoader.next_batch` slices token stream locally (no MPI sharding here).
- Training loop:
  - Forward: `gpt_forward_with_loss` → `logits`, `loss`.
  - Backward: `backward(loss)`; clear activations.
  - Optional `mpi_allreduce_grads` sums grads across ranks (data parallel).
  - Optimizer: `adam_step` (with optional grad clipping) → `adam_zero_grad`.
  - Master rank prints loss.
- Master rank saves weights (`save_weights("trained_weights.bin")`).
- Text generation (master only): sliding window into `gpt_forward_logits`, greedy pick via `greedy_select_next_token`; stop on EOS or length cap; print decoded tokens.

## Model Stack (CUDA-backed ops, GPT-2 topology)
- Token IDs `(B,T)` → `wte` token embedding; `wpe` position embedding; `broadcast_add` → `x0 (B,T,n_embd)`.
- Repeat `n_layer` times:
  - `LN1(x)` → Multi-Head Attention (per head `q/k/v` linears; scaled dot-product with causal mask; softmax; value weighted sum; concat heads; projection).
  - Residual: `x + attn_out` → `x1`.
  - `LN2(x1)` → `MLP` (`c_fc: n_embd→4*n_embd` → `GELU` → `c_proj: 4*n_embd→n_embd`).
  - Residual: `x1 + mlp_out` → next `x`.
- Final `LN_f` → `lm_head Linear: n_embd→vocab_size` → `logits (B,T,V)` (drives CE loss).
