# GPT-2 Architecture Notes

## Serial (stage0) — Data Flow Modules
- Tokenizer (`tokenizer.h`): reads `data/dummy.txt`, extracts tokens, builds vocab, encodes ids, pads with `[PAD]`/`[EOS]`, exposes vocab size and encoded buffer.
- DataLoader (`dataloader.h`): walks the encoded buffer, emits `inputs`/`targets` of shape `batch_size*seq_len` with wraparound.
- Model front (`gpt.h`): `gpt_forward_with_loss` returns logits and cross-entropy loss; `gpt_forward_logits` for generation.
- Autograd (`autograd.h` with `tensor.h` tracker): builds a topological order over tracked tensors and calls `_backward` callbacks.
- Optimizer (`adam.h`): Adam update with optional gradient clipping and linear learning rate decay; zeroes gradients each step.
- Checkpointing (`checkpoint.h`): binary save/load of all parameters with shape validation.
- Generation: sliding window into `gpt_forward_logits`, greedy next-token selection, prints decoded tokens.

## Serial (stage0) — Model Stack Modules
- Embeddings: token embedding `wte` and position embedding `wpe`, combined via `broadcast_add`.
- Transformer blocks (repeat `n_layer`):
  - `LayerNorm` → Multi-Head Attention: per-head q/k/v linears, scaled dot-product with causal mask, softmax, weighted value sum, head concatenation, projection linear → residual add.
  - `LayerNorm` → MLP: `c_fc` (expand), GELU, `c_proj` (project) → residual add.
- Output head: final `LayerNorm` → `lm_head` linear → logits over vocab.

## Differences by Stage
- Stage 1:
  - MPI everywhere: `MPI_Init`/`MPI_Finalize`, `MPI_Allreduce` for loss and gradients, and `MPI_Bcast` for tokenizer/dataloader/params.
  - Batch sharding: global batch divided by `world_size`; dataloader offsets per rank.
  - Param sync metadata (`ParamSyncInfo`) to track shared tensors; shared grads scaled by `1/world_size` after allreduce.
  - Distributed generation: logits averaged across ranks; rank 0 performs greedy decode and prints.
  - `gpt_set_distributed` enables distributed-aware matmul/softmax/gelu/xent (still CPU compute).

- Stage2 CUDA (GPU compute, optional MPI) 
  CUDA device selection per rank; matmul, softmax, GELU, layernorm run via CUDA kernels; rank 0 can broadcast initial params; gradients optionally allreduced; each rank tokenizes locally; same graph as serial but executed on GPU.
  - CUDA device selection per rank; core ops offloaded to CUDA kernels (`cuda_kernels.cu`, `cuda_utils.h`).
  - Optional MPI: rank 0 can broadcast params; grads optionally reduced via `mpi_allreduce_grads`.
  - No tokenizer/dataloader broadcast—each rank reads and tokenizes locally; sync occurs only through parameters/gradients.
  - Same model graph, but execution uses GPU-backed kernels; data parallelism is optional and lighter-weight.

- Stage3 MPI+CUDA (GPU data parallel): 
  rank-to-GPU mapping, tokenizer built on rank 0 then broadcast; dataloader sharded by rank/world_size; parameters broadcast; gradients and losses allreduced via MPI; distributed configuration applied to GPU-backed ops for consistent data-parallel training.
  - Combined MPI + CUDA: each rank binds to a GPU (`rank % device_count`) and owns a shard of the global batch.
  - Tokenizer built on rank 0 and broadcast; dataloader sharded by rank/world_size.
  - Parameters broadcast from rank 0; gradients reduced with `mpi_allreduce_grads`; losses averaged via `mpi_allreduce_loss`.
  - `gpt_set_distributed` and `mpi_utils.h` coordinate distributed settings across GPU-backed ops.
  - End-to-end flow mirrors stage2 CUDA compute but with mandatory data-parallel synchronization across ranks.
