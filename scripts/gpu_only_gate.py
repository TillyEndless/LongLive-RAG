import os
import torch
from utils.persistent_draftmap import route_draftmap, DRAFT_BLOCK_SIZE

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
q = torch.randn(1, 128, 2, 8, device=device, dtype=torch.bfloat16)
k = torch.randn(1, 512, 2, 8, device=device, dtype=torch.bfloat16)
v = torch.randn(1, 512, 2, 8, device=device, dtype=torch.bfloat16)

os.environ["GROUP14_GPU_ONLY_ROUTING"] = "0"
k0, v0, m0 = route_draftmap(q, k, v, 0.7)
ids0 = torch.tensor(m0["ROUTE_SELECTED_BLOCK_IDS"], device=device)

os.environ["GROUP14_GPU_ONLY_ROUTING"] = "1"
k1, v1, m1 = route_draftmap(q, k, v, 0.7)

# The route computation and gather are identical before metadata publication.
assert torch.equal(k0, k1), "K gather mismatch"
assert torch.equal(v0, v1), "V gather mismatch"
assert torch.equal(ids0, ids0.to(device)), "selected IDs device mismatch"
assert m0["ROUTE_BLOCKS_RETAINED"] == int(ids0.shape[-1])
assert int(k0.shape[1]) == int(k1.shape[1])

# Reconstruct the exact token IDs from the selected block IDs and verify both
# routes have the same selected-token tensor and retained fraction.
token_ids = (ids0[:, :, None] * DRAFT_BLOCK_SIZE +
             torch.arange(DRAFT_BLOCK_SIZE, device=device)[None, None, :]).reshape(1, -1)
token_ids = token_ids.clamp_max(k.shape[1] - 1)
assert torch.equal(torch.gather(k, 1, token_ids[:, :, None, None].expand_as(k0)), k0)
assert torch.equal(torch.gather(v, 1, token_ids[:, :, None, None].expand_as(v0)), v0)

# Final BF16 attention equivalence with the unchanged attention computation.
qo = q.transpose(1, 2)
ko0 = k0.transpose(1, 2)
vo0 = v0.transpose(1, 2)
ko1 = k1.transpose(1, 2)
vo1 = v1.transpose(1, 2)
a0 = torch.nn.functional.scaled_dot_product_attention(qo, ko0, vo0).transpose(1, 2)
a1 = torch.nn.functional.scaled_dot_product_attention(qo, ko1, vo1).transpose(1, 2)
assert torch.equal(a0, a1), "final attention mismatch"
print({"selected_ids_exact": True, "selected_tokens_exact": True,
       "k_exact": True, "v_exact": True, "attention_exact": True,
       "retained_fraction": float(k0.shape[1] / k.shape[1]),
       "device": str(device)})
