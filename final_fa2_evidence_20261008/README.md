# FA2 final candidate and validation provenance — 2026-10-08

This archive preserves two DISTINCT source snapshots. Do not attribute verified measured speedups to a later modified source snapshot without a repeat validation.

Validation report: validation/h200_real_overlap_validation.md
Canonical H200 validation: B01 120 latent frames; serial safe 93.031s vs grouped streaming G4 88.019s (5.39% faster); same-call attention max_abs=0. Validation report lists source SHA256 177fdc3ffe68d6e4ca76b49659dc1fc3f421775485dd3772b44d56d838b0a2a5, which was NOT found among inspected current snapshots.
Validated extension SHA256 listed in report: cec3f4625bc551ec1af846789234ae2720fa0a89aa686017336046c524ccfdba. This matches the .so archived on Hugging Face.
Validation pipeline SHA256 ebb80348d3492db701eaa4cb3286c63f6519c99a2579f91a91c46b9076a7bca2, matches post_validation_candidate/causal_inference.py.

formal_bundle_source/ is a distinct BBAO4 formal snapshot (causal_model_latentmem.py SHA babac411c7209ac490d99ea1e92c14985222fddbf320067d8d5bd242dc388ec9), NOT the source SHA shown in the real-overlap validation report.
post_validation_candidate/ has newer current source (SHA 9d40fb295d43d656af466f87632457d23a52626b0731473fefa5868906d4c4f9), NOT byte-identical to validation report source.

This archive provides independently recoverable candidates and the measured evidence; full exact build-source match remains an explicit pending audit item.
