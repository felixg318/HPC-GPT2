# Stage0 Serial GPT-2 Architecture

Reference: `stage0_serial/train_gpt2.cpp` and its headers (`gpt.h`, `block.h`, `multihead_attention.h`, `head.h`, `mlp.h`, `layernorm.h`, `linear.h`, `softmax.h`, `matmul.h`, `dataloader.h`, `tokenizer.h`, `adam.h`, `autograd.h`, `checkpoint.h`).

## Unique Traits
- Single-process, CPU-only execution; no MPI/CUDA; all tensors live on host.
- Autograd and optimizer operate purely in-process; no gradient communication.

## Data Flow
- `data/dummy.txt` → `Tokenizer` (`tokenizer_extract`, `tokenizer_encode`, `tokenizer_pad_to` adds `[PAD]`/`[EOS]`).
- `DataLoader.next_batch` → `inputs[B*T]`, `targets[B*T]` sliding through token stream.
- `gpt_forward_with_loss` → `logits(B,T,V)`, `cross_entropy_loss_3d` → `loss(1)`.
- `backward(loss)` traverses tracked tensors.
- `adam_step` (with optional `clip_grad_norm` and `linear_lr_decay`) updates collected parameters, then `adam_zero_grad`.
- Loop for `epochs`, then `save_weights("trained_weights.bin")`.
- Demo: `gpt_forward_logits` on a prompt window → `greedy_select_next_token` loop prints generated text.

## Model Stack (GPT-2 style)
- Input token IDs `(B,T)`:
  - Token embedding `wte: vocab_size → n_embd`
  - Position embedding `wpe: T → n_embd`
  - `broadcast_add` → `x0 (B,T,n_embd)`
- Repeat for `n_layer` blocks:
  - `LN1(x)` → `MultiHeadAttention`:
    - Per head: `q = Linear_q(x)`, `k = Linear_k(x)`, `v = Linear_v(x)`
    - `attn_logits = (q @ k^T) / sqrt(head_size)`, causal mask (`t_k > t_q → -1e30`)
    - `attn_probs = softmax(attn_logits)`
    - `head_out = attn_probs @ v`
    - Concatenate heads → `concat (B,T,n_embd)` → final `Linear proj` → `attn_out`
  - Residual: `x + attn_out` → `x1`
  - `LN2(x1)` → `MLP` (`c_fc: n_embd→4*n_embd` → `GELU` → `c_proj: 4*n_embd→n_embd`)
  - Residual: `x1 + mlp_out` → next `x`
- Final: `LN_f(x)` → `lm_head Linear: n_embd → vocab_size` → `logits (B,T,V)`
