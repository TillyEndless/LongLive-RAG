# Sparse-ratio configuration archive

The active corrected Group14/15 sweep is now defined by retained interaction
ratios `0.70`, `0.50`, `0.30`, and `0.60`, represented by removal labels
`sparse30`, `sparse50`, `sparse70`, and `sparse40` respectively.

The pre-existing `configs/group12_15_corrected_campaign/` files named
`sparse05`, `sparse10`, `sparse20`, and `sparse30` are retained as legacy
configuration history. They are not deleted or overwritten, and they are not
admitted by the active campaign runners. The active preparation script derives
new snapshots from the `sparse30` template and rewrites the runtime field to
the authoritative retained ratio.
