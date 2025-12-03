# Stage 0 Serial GPT-2 Overview

This directory hosts a from‑scratch, single‑process GPT‑2 style implementation built out of small C headers. The files are meant to be composed together inside `train_gpt2.cpp` and `inference.cpp`, which exercise the model for training and generation respectively. Below is a guide to every file and the role it plays in the end‑to‑end pipeline.

## Core Tensor & Autograd
- `tensor.h` – minimal n‑dimensional tensor with data/grad buffers, shape metadata, and intrusive autograd hooks via `_inputs` and `_backward`.
- `autograd.h` – tracks computation graphs, performs topological sort (`build_topo`) and backprop (`backward`) by calling each tensor’s registered `_backward`.

## Linear Algebra & Math Ops
- `matmul.h` – forward/backward for 2D and batched 3D matrix multiplications, used heavily in attention and MLP layers.
- `add.h` – elementwise addition helper plus residual gradient wiring.
- `transpose.h` – utility for reshaping/permutes required in attention weight handling.
- `gelu.h` – Gaussian Error Linear Unit activation used in the MLP block.
- `softmax.h` – numerically stable softmax (2D/3D) with backwards, powering attention probability computation.
- `layernorm.h` – LayerNorm forward/backward for 2D/3D tensors, with learnable gamma/beta parameters.

## Model Building Blocks
- `embedding.h` – token and positional embedding layers; look up table forward/backward.
- `linear.h` – fully connected layer implementation with optional bias.
- `head.h` – single self-attention head (QKV projections, masked softmax, value aggregation).
- `multihead_attention.h` – aggregates many `Head` instances, concatenates their outputs, and applies the final projection.
- `mlp.h` – two-layer feed-forward network (Linear → GELU → Linear) used inside each transformer block.
- `block.h` – full transformer block: LN → MHA → residual, then LN → MLP → residual.
- `gpt.h` – top-level GPT-2 model: embeddings, stack of `Block`s, final LayerNorm, LM head, plus helpers for logits/loss.

## Training Utilities
- `tokenizer.h` – byte-pair tokenizer clone: reads corpus (`dummy_data.txt`), builds vocab, encodes/decodes tokens.
- `dataloader.h` – turns token streams into sequential batches (inputs/targets) for training loops.
- `cross_entropy.h` – cross-entropy loss module with backward pass for logits vs. one-step targets.
- `adam.h` – Adam optimizer implementation with bias correction, optional LR scheduler, and gradient norm clipping helper.
- `checkpoint.h` – save/load model weights to `trained_weights.bin`.
- `broadcast.h` – tiny helper for tensor serialization when sharing parameters (unused in pure serial stage but exists for parity with MPI stage).

## Executables
- `train_gpt2.cpp` – orchestrates tokenizer prep, model instantiation, training loop, optimizer step, periodic loss logging, and post-training text generation demo.
- `inference.cpp` – loads checkpoints and runs a pure generation loop for inspection; the `inference` binary is the compiled artifact.

## Data Files
- `dummy_data.txt` and `dummy_data2.txt` – small sample corpora used for tokenizer demonstration and training sanity checks.
- `trained_weights.bin` – latest serialized weights produced by training (`checkpoint.h` format); useful for `inference`.

## Build Artifacts
- `train_gpt2` and `inference` binaries – compiled outputs of their respective `.cpp` sources (kept here for convenience, but can be regenerated via `g++`/`clang++`).

Taken together, these components mirror the GPT‑2 architecture: tokenizer → embeddings → repeated transformer blocks (attention + MLP) → final layernorm + LM head, with autograd and Adam enabling gradient-based training. Use `train_gpt2.cpp` for end-to-end runs and `inference.cpp` for sampling with saved checkpoints.

## Code Flow (train_gpt2.cpp)
- Parse optional `--seed`, set RNG seed.
- Build tokenizer on `dummy_data.txt`, pad corpus to at least one full batch+1 target token, and extract vocab size.
- Initialize GPT model (`gpt_init`), collect parameters, and create Adam optimizer with linear LR decay.
- Create dataloader over encoded tokens and log expected token counts.
- Training loop: get batch → forward (`gpt_forward_with_loss`) → backward → optimizer step → zero grads → log loss.
- After training: save weights (`checkpoint.h`), run greedy text generation demo, print timing, and free all resources.
