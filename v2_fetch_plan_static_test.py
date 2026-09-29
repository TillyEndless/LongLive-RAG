"""Deterministic semantic checks for the unified v2 fetch plan."""
import torch
from utils.attention_fetch_plan import make_plan, submit, wait
from utils.local_lowbit_kv import LocalLowbitKVStore


def main():
    local = {i: (torch.full((1, 2, 1, 2), float(i), dtype=torch.bfloat16),
                 torch.full((1, 2, 1, 2), float(i) + .5, dtype=torch.bfloat16))
             for i in range(1, 5)}
    history = {i: (torch.full((1, 2, 1, 2), float(i), dtype=torch.bfloat16),
                   torch.full((1, 2, 1, 2), float(i) + .25, dtype=torch.bfloat16))
               for i in (2, 4)}
    plan = make_plan([2, 4], [1, 3], history, local)
    handle = wait(submit(plan, torch.device("cpu")), torch.device("cpu"))
    assert plan.history_ids == [2, 4]
    assert plan.promotion_ids == [1, 3]
    assert list(handle.history) == [2, 4]
    assert list(handle.promotion) == [1, 3]
    assert set(plan.history_ids).isdisjoint(plan.promotion_ids)
    assert handle.history_bytes > 0 and handle.promotion_bytes > 0
    assert handle.physical_copy_count == 8

    store = LocalLowbitKVStore(4, 2, 1, 2, "int8_fp8", torch.device("cpu"))
    k = torch.ones((1, 8, 1, 2), dtype=torch.bfloat16)
    store.insert_frames(0, k, k)
    before = store.persistent_bytes()
    selected = store.select_top([0., 1., 0., 1.], .5)
    assert selected == [3, 1]
    assert store.persistent_bytes() == before
    print("GROUP12_15_V2_FETCH_PLAN_STATIC_TESTS=PASS")


if __name__ == "__main__":
    main()
