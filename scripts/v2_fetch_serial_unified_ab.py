"""Small CUDA AB test for v2 fetch scheduling (no model inference)."""
import json
import os
import time
import torch
from utils.attention_fetch_plan import make_plan, submit, wait


def one(mode, sources_h, sources_p):
    device = torch.device("cuda")
    ids_h, ids_p = [1, 3], [2, 4]
    plan = make_plan(ids_h, ids_p, sources_h, sources_p)
    if mode == "serial":
        t0 = time.perf_counter()
        out_h, out_p = {}, {}
        for i in ids_h:
            out_h[i] = tuple(x.to(device) for x in sources_h[i])
        torch.cuda.synchronize()
        for i in ids_p:
            out_p[i] = tuple(x.to(device) for x in sources_p[i])
        torch.cuda.synchronize()
        exposed = time.perf_counter() - t0
        handle = type("H", (), {"history": out_h, "promotion": out_p,
                                "history_bytes": sum(x[0].numel()*2+x[1].numel()*2 for x in sources_h.values()),
                                "promotion_bytes": sum(x[0].numel()*2+x[1].numel()*2 for x in sources_p.values())})()
    else:
        handle = submit(plan, device)
        t0 = time.perf_counter(); wait(handle, device); exposed = time.perf_counter() - t0
    torch.cuda.synchronize()
    assembled_h = torch.cat([handle.history[i][0] for i in ids_h], 1)
    assembled_p = torch.cat([handle.promotion[i][0] for i in ids_p], 1)
    return handle, exposed, torch.cat((assembled_h, assembled_p), 1)


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    out = {"backend": "pytorch_h2d_only", "native_anemoi_kernel": False}
    for label in ("group14_v2", "group15_v2"):
        sources_h = {i: (torch.randn(1, 64, 12, 128, dtype=torch.bfloat16),
                        torch.randn(1, 64, 12, 128, dtype=torch.bfloat16)) for i in (1, 3)}
        sources_p = {i: (torch.randn(1, 64, 12, 128, dtype=torch.bfloat16),
                        torch.randn(1, 64, 12, 128, dtype=torch.bfloat16)) for i in (2, 4)}
        hs, serial_s, serial = one("serial", sources_h, sources_p)
        hu, unified_s, unified = one("unified", sources_h, sources_p)
        out[label] = {
            "serial_exposed_fetch_s": serial_s,
            "unified_exposed_fetch_s": unified_s,
            "same_layout": list(serial.shape) == list(unified.shape),
            "max_abs_error": float((serial - unified).abs().max().item()),
            "numerically_equal": bool(torch.equal(serial, unified)),
            "unified_submission": True,
            "native_kernel": False,
        }
    with open(os.environ.get("V2_FETCH_AB_OUTPUT", "results/v2_fetch_serial_unified_ab.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
