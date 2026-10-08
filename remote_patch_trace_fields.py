from pathlib import Path
p=Path('/data/zxl/LongLive-RAG-group11_15_h200/wan/modules/causal_model_latentmem.py')
s=p.read_text()
old='''                            archive_materialized_meta.update(q_sparse_meta)
                            archive_materialized_meta["FINAL_HISTORY_IDS"] = retained_ids
'''
new='''                            archive_materialized_meta.update(q_sparse_meta)
                            archive_materialized_meta["Q_SPARSE_INPUT_IDS"] = selected_ids
                            archive_materialized_meta["DRAFTMAP_IMPORTANCE"] = {str(k): float(v) for k, v in score_map.items()}
                            archive_materialized_meta["FINAL_HISTORY_IDS"] = retained_ids
'''
if old not in s:
    raise SystemExit("anchor missing")
p.write_text(s.replace(old,new,1))

