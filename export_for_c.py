# export_for_c.py
import torch

data = torch.load("gpt_debug_forward.pt", map_location="cpu")
dbg = data["block0_head0_debug"]
state = data["state_dict"]
config = data["config"]

x_in = dbg["x_in"]   # (1, 5, 16)
q_ref = dbg["q"]     # (1, 5, 8)
k_ref = dbg["k"]     # (1, 5, 8)
v_ref = dbg["v"]     # (1, 5, 8)
att_ref = dbg["att"] # (1, 5, 5)
out_ref = dbg["out"] # (1, 5, 8)

n_embd = config["n_embd"]       # 16
n_head = config["n_head"]       # 2
head_size = n_embd // n_head    # 8

# Head 0 weights from PyTorch
Wq = state["transformer.h.0.attn.heads.0.query.weight"]  # usually (8,16)
Wk = state["transformer.h.0.attn.heads.0.key.weight"]
Wv = state["transformer.h.0.attn.heads.0.value.weight"]

# Convert to C layout: (in_dim, out_dim) = (16, 8)
def to_c_weight(W):
    if W.shape == (head_size, n_embd):
        return W.t().contiguous()
    elif W.shape == (n_embd, head_size):
        return W.contiguous()
    else:
        raise ValueError(f"Unexpected weight shape: {W.shape}")

Wq_c = to_c_weight(Wq)
Wk_c = to_c_weight(Wk)
Wv_c = to_c_weight(Wv)

# Self-check: q = x_in @ Wq_c must match dbg["q"]
x_flat = x_in[0]   # (T, C)
q_flat = q_ref[0]  # (T, head_size)

q_check = x_flat @ Wq_c          # (5, 8)
max_diff = (q_check - q_flat).abs().max().item()
print(f"// sanity: max diff between q_check and q_ref (Python) = {max_diff:.9e}")

if max_diff > 1e-5:
    raise RuntimeError("Wq_c layout is wrong; q doesn't match dbg['q']")

def to_c_array(name, tensor):
    tensor = tensor.contiguous().view(-1)
    elems = ", ".join(f"{x.item():.9f}" for x in tensor)
    print(f"// {name}, numel={tensor.numel()}")
    print(f"float {name}[] = {{ {elems} }};\n")

# Now actually dump everything as C arrays:
to_c_array("x_in_ref", x_in)
to_c_array("q_ref", q_ref)
to_c_array("k_ref", k_ref)
to_c_array("v_ref", v_ref)
to_c_array("att_ref", att_ref)
to_c_array("out_ref", out_ref)

to_c_array("Wq_ref", Wq_c)
to_c_array("Wk_ref", Wk_c)
to_c_array("Wv_ref", Wv_c)
