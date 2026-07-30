# Provenance

`provenance/stage512a_v7_core.sha256` and
`provenance/data_analysis_core.sha256` bind every imported file to its source
bytes. `provenance/compact_docs.sha256` binds the compact documentation input.
`provenance/CURATION_RECEIPT.json` is a builder preparation receipt: it binds
the pre-commit tree definition, source release identity, policy, builder,
exact Git base, and intended review branch. A later controller may commit that
tree or add a local archive tag; those later actions are intentionally outside
the receipt scope and require separate evidence.
