# Data pipeline

The current runtime consumes the weighted offline plan through
`training/dataline_offline/` while reusing safety and decoding components from
`training/dataline/`. Keep the full `training/` import closure.

Stage512b-real must use a new, reviewed data identity. Real images must not be
upsampled; the Stage512a weighted exception binding is historical and must not
be generalized. Generated plans, catalogs, source manifests, and data assets
remain outside Git.
