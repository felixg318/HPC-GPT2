# export_gpt_full.py
import json
from pathlib import Path
import types

import torch
from dataclasses import asdict


def load_notebook_module(nb_path):
    """
    Execute code cells from a .ipynb file inside a synthetic module and return it.

    We stop execution once GPT/GPTConfig appear to avoid running demo cells that
    instantiate the model or write files.
    """
    nb_path = Path(nb_path)
    nb = json.loads(nb_path.read_text())
    module = types.ModuleType(nb_path.stem)
    module.__file__ = str(nb_path)
    module.__dict__.setdefault("__builtins__", __builtins__)

    target_symbols = {"GPT", "GPTConfig"}
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if not source.strip():
            continue
        code = compile(source, filename=f"{nb_path}", mode="exec")
        exec(code, module.__dict__)
        if target_symbols.issubset(module.__dict__):
            break

    missing = target_symbols - module.__dict__.keys()
    if missing:
        raise ImportError(
            f"Could not find symbols {missing} in notebook {nb_path}"
        )
    return module


nb_module = load_notebook_module(Path(__file__).with_name("gpt2.ipynb"))
GPT = nb_module.GPT
GPTConfig = nb_module.GPTConfig

# Rebuild the tiny debug config
torch.manual_seed(0)

config = GPTConfig(
    block_size=8,
    vocab_size=50,
    n_layer=2,
    n_head=2,
    n_embd=16,
    dropout=0.0,
)

model = GPT(config)
model.eval()

# Fixed tiny input
idx = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)  # (B=1,T=5)
with torch.no_grad():
    logits, loss = model(idx, targets=idx)

state = model.state_dict()
cfg = asdict(config)

B, T = idx.shape
vocab_size = cfg["vocab_size"]
block_size = cfg["block_size"]
n_layer = cfg["n_layer"]
n_head = cfg["n_head"]
n_embd = cfg["n_embd"]
head_size = n_embd // n_head
hidden_dim = 4 * n_embd  # GPT-2 MLP size

# --------------------- Helper functions ---------------------


def to_c_array(name, tensor):
    tensor = tensor.contiguous().view(-1)
    elems = ", ".join(f"{x.item():.9f}" for x in tensor)
    print(f"// {name}, numel={tensor.numel()}")
    print(f"float {name}[] = {{ {elems} }};\n")


def to_c_int_array(name, tensor):
    tensor = tensor.contiguous().view(-1)
    elems = ", ".join(str(int(x.item())) for x in tensor)
    print(f"// {name}, numel={tensor.numel()}")
    print(f"int {name}[] = {{ {elems} }};\n")


def to_c_weight(W, in_dim, out_dim):
    """
    Convert PyTorch Linear weight to C layout (in_dim, out_dim).
    PyTorch: (out_dim, in_dim)
    C:       (in_dim, out_dim)
    """
    if W.shape == (out_dim, in_dim):
        return W.t().contiguous()
    elif W.shape == (in_dim, out_dim):
        return W.contiguous()
    else:
        raise ValueError(
            f"Unexpected Linear weight shape {W.shape}, "
            f"expected ({out_dim},{in_dim}) or ({in_dim},{out_dim})"
        )


# --------------------- Embeddings ---------------------

wte_weight = state["transformer.wte.weight"]  # (vocab_size, n_embd)
wpe_weight = state["transformer.wpe.weight"]  # (block_size, n_embd)

to_c_array("wte_weight_ref", wte_weight)
to_c_array("wpe_weight_ref", wpe_weight)

# --------------------- Per-layer parameters ---------------------

# LayerNorms across all layers
ln1_gamma_all = []
ln1_beta_all = []
ln2_gamma_all = []
ln2_beta_all = []

# Attention Q/K/V and proj
Wq_all = torch.zeros(n_layer, n_head, n_embd, head_size)
Wk_all = torch.zeros_like(Wq_all)
Wv_all = torch.zeros_like(Wq_all)

Wproj_all = torch.zeros(n_layer, n_embd, n_embd)
bproj_all = torch.zeros(n_layer, n_embd)

# MLP weights
Wfc_all = torch.zeros(n_layer, n_embd, hidden_dim)
bfc_all = torch.zeros(n_layer, hidden_dim)
Wproj_mlp_all = torch.zeros(n_layer, hidden_dim, n_embd)
bproj_mlp_all = torch.zeros(n_layer, n_embd)

for l in range(n_layer):
    # LN1, LN2
    ln1_gamma = state[f"transformer.h.{l}.ln_1.weight"]  # (n_embd,)
    ln1_beta = state[f"transformer.h.{l}.ln_1.bias"]
    ln2_gamma = state[f"transformer.h.{l}.ln_2.weight"]
    ln2_beta = state[f"transformer.h.{l}.ln_2.bias"]

    ln1_gamma_all.append(ln1_gamma)
    ln1_beta_all.append(ln1_beta)
    ln2_gamma_all.append(ln2_gamma)
    ln2_beta_all.append(ln2_beta)

    # Heads: Q, K, V
    for h in range(n_head):
        Wq = state[f"transformer.h.{l}.attn.heads.{h}.query.weight"]
        Wk = state[f"transformer.h.{l}.attn.heads.{h}.key.weight"]
        Wv = state[f"transformer.h.{l}.attn.heads.{h}.value.weight"]

        Wq_c = to_c_weight(Wq, n_embd, head_size)
        Wk_c = to_c_weight(Wk, n_embd, head_size)
        Wv_c = to_c_weight(Wv, n_embd, head_size)

        Wq_all[l, h] = Wq_c
        Wk_all[l, h] = Wk_c
        Wv_all[l, h] = Wv_c

    # Attention proj
    Wproj = state[f"transformer.h.{l}.attn.proj.weight"]  # (n_embd, n_embd)
    bproj = state[f"transformer.h.{l}.attn.proj.bias"]    # (n_embd,)

    Wproj_c = to_c_weight(Wproj, n_embd, n_embd)          # (n_embd, n_embd)
    Wproj_all[l] = Wproj_c
    bproj_all[l] = bproj

    # MLP: c_fc (n_embd -> hidden_dim), c_proj (hidden_dim -> n_embd)
    Wfc = state[f"transformer.h.{l}.mlp.c_fc.weight"]       # (hidden_dim, n_embd)
    bfc_l = state[f"transformer.h.{l}.mlp.c_fc.bias"]       # (hidden_dim,)
    Wproj_mlp = state[f"transformer.h.{l}.mlp.c_proj.weight"]  # (n_embd, hidden_dim)
    bproj_mlp_l = state[f"transformer.h.{l}.mlp.c_proj.bias"]  # (n_embd,)

    Wfc_c = to_c_weight(Wfc, n_embd, hidden_dim)            # (n_embd, hidden_dim)
    Wproj_mlp_c = to_c_weight(Wproj_mlp, hidden_dim, n_embd)

    Wfc_all[l] = Wfc_c
    bfc_all[l] = bfc_l
    Wproj_mlp_all[l] = Wproj_mlp_c
    bproj_mlp_all[l] = bproj_mlp_l

ln1_gamma_all = torch.stack(ln1_gamma_all, dim=0)  # (n_layer, n_embd)
ln1_beta_all = torch.stack(ln1_beta_all, dim=0)
ln2_gamma_all = torch.stack(ln2_gamma_all, dim=0)
ln2_beta_all = torch.stack(ln2_beta_all, dim=0)

to_c_array("ln1_gamma_all_ref", ln1_gamma_all)
to_c_array("ln1_beta_all_ref", ln1_beta_all)
to_c_array("ln2_gamma_all_ref", ln2_gamma_all)
to_c_array("ln2_beta_all_ref", ln2_beta_all)

to_c_array("Wq_all_ref", Wq_all)
to_c_array("Wk_all_ref", Wk_all)
to_c_array("Wv_all_ref", Wv_all)

to_c_array("Wproj_all_ref", Wproj_all)
to_c_array("bproj_all_ref", bproj_all)

to_c_array("Wfc_all_ref", Wfc_all)
to_c_array("bfc_all_ref", bfc_all)
to_c_array("Wproj_mlp_all_ref", Wproj_mlp_all)
to_c_array("bproj_mlp_all_ref", bproj_mlp_all)

# --------------------- Final LayerNorm + LM head ---------------------

ln_f_gamma = state["transformer.ln_f.weight"]  # (n_embd,)
ln_f_beta  = state["transformer.ln_f.bias"]
to_c_array("ln_f_gamma_ref", ln_f_gamma)
to_c_array("ln_f_beta_ref", ln_f_beta)

lm_head_W = state["lm_head.weight"]           # (vocab_size, n_embd) or (out,in)
# But in GPT-2 style, lm_head is (vocab_size, n_embd). We want (n_embd, vocab_size).
lm_head_W_c = to_c_weight(lm_head_W, n_embd, vocab_size)
to_c_array("lm_head_weight_ref", lm_head_W_c)

# --------------------- Input, logits, loss ---------------------

to_c_int_array("idx_ref", idx)        # shape (1,5)
to_c_array("logits_ref", logits)      # shape (1,5,vocab_size)

print("// loss_ref (scalar)")
print(f"float loss_ref = {loss.item():.9f};\n")
