# Stage1 MPI GPT-2 Architecture

Reference: `stage1_mpi/train_gpt2.cpp` and headers (`gpt.h`, `block.h`, `multihead_attention.h`, `head.h`, `mlp.h`, `layernorm.h`, `linear.h`, `softmax.h`, `matmul.h`, `dataloader.h`, `tokenizer.h`, `adam.h`, `autograd.h`, `checkpoint.h`).

## Unique Traits
- Pure MPI data parallelism on CPU: global batch split across ranks; gradients summed with `MPI_Allreduce`.
- Parameter sync metadata (`ParamSyncInfo`) marks shared vs per-rank tensors; shared params scaled by `1/world_size` after reduction.
- Tokenizer and dataloader objects are broadcast from rank 0, keeping vocab and sampling state consistent.
- Generation averages logits across ranks before greedy decode; rank 0 drives token selection.

## Data Flow
- MPI init (`MPI_Init`); parse `--seed`; per-rank RNG via `tensor_set_seed`.
- Rank 0: `Tokenizer` over `../data/dummy.txt` → `tokenizer_extract` → `tokenizer_encode` → `tokenizer_pad_to`; broadcast tokenizer (`tokenizer_broadcast`). All ranks build vocab/encoded buffers.
- Per-rank batch size = `global_batch_size / world_size`; `DataLoader.init_with_tokenizer` then `dataloader_broadcast`; `dataloader_next_batch` slices the token stream with rank offsets.
- Model init `gpt_init` + `gpt_set_distributed` (enables distributed matmul/softmax/gelu/xent settings); collect params.
- Build `ParamSyncInfo` to tag shared vs per-rank tensors; rank 0 broadcasts shared params so all ranks start from identical weights.
- Training loop:
  - Forward: `gpt_forward_with_loss` → `logits(B,T,V)`, `loss(1)`; `MPI_Allreduce` on loss for reporting.
  - Backward: `backward(loss)`; gather all param grads into a contiguous buffer; `MPI_Allreduce` to sum; copy back, scaling shared params by `1/world_size`.
  - Optimizer: `adam_step` (with optional `clip_grad_norm` and `linear_lr_decay`) → `adam_zero_grad`.
  - Rank 0 prints epoch loss.
- Rank 0 saves weights (`save_weights("trained_weights.bin")`).
- Distributed generation: for each step `gpt_forward_logits` → logits averaged across ranks (`synchronize_logits_across_ranks`); rank 0 picks next token, broadcasts; loop until max tokens or EOS; rank 0 prints decoded text.

## Model Stack (same topology as stage0)
- Token IDs `(B,T)` → token embedding `wte` and position embedding `wpe`; `broadcast_add` → `x0 (B,T,n_embd)`.
- Repeat `n_layer` times:
  - `LN1(x)` → Multi-Head Attention (per head `q/k/v` linears; scaled dot-product with causal mask; softmax; value weighted sum; concat heads; projection).
  - Residual: `x + attn_out` → `x1`.
  - `LN2(x1)` → `MLP` (`c_fc: n_embd→4*n_embd` → `GELU` → `c_proj: 4*n_embd→n_embd`).
  - Residual: `x1 + mlp_out` → next `x`.
- Final `LN_f` → `lm_head Linear: n_embd→vocab_size` → `logits (B,T,V)` (fed to CE loss during training).
