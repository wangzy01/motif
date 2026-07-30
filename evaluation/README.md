# Evaluation

`benchmarks/` contains reusable benchmark assets and scorers; `runners/`
contains the explicitly selected portable entry points.
`reference/stage512a_final_20260728/` is a byte-exact protocol snapshot whose
absolute paths are historical evidence, not portable launch entry points.
New comparisons must preserve raw prompts, disclose prompt rewriting, and use
the same public protocol across models.
