# Calibration specimens

`specimens_calib.csv` lists 89 physical specimens with measured envelope
dimensions (cm), empty mass (g) and, for 61 of them, the mass after filling
with water, a metal weight or a stone (`mass_loaded_g`). No photograph of any
of these specimens appears in real495 or in Hybrid-v3.3: they are the specimens
of the PackPhys collection (same acquisition protocol, same authors) that were
photographed on other frames. real495 itself contributes 25 specimens, listed
with their ids in `../real495/measurements_specimens.csv`; the two sets are
disjoint by construction and the union is the whole PackPhys catalogue (114).

Everything the physics channel needs is estimated from this table alone
(`scripts/calibration/build_calibration.py` -> `configs/calibration_packphys.yaml`):

| quantity | how |
|---|---|
| `densities[c]` | median of empty mass / envelope volume over the specimens of class c |
| `density_sigma[c]` | s.d. of log density between specimens of c (pooled median for a single specimen) |
| `specimens[c]` | per-specimen density and loaded/empty ratio, drawn by the tuple generator |
| `volume_scale[src]` | one factor per image source that puts the bounding-box volume proxy in cm^3, fitted on class medians |
| `volume_sigma[c]` | residual log-s.d. of the proxy within a class, measured on the photographed specimens |

## Specimen-disjoint folds

`folds/` (and `folds_rep{1,2,3}/`, three further random partitions) split the
89 specimens into four class-stratified groups (`manifest.json`). In fold k the
group k is the test pool and the other three form the development pool. The
development pool gives the density table, the volume scales, the training and
validation masses (each item against the table without its own specimen) and
the decision thresholds; the test pool gives the test masses only. Every
specimen is a test specimen in exactly one fold of each partition.

```bash
python scripts/calibration/build_calibration.py \
    --photos-coco <PackPhys instances_export.json> \
    --extra-coco beach=datasets/processed_ood/beach_litter_v2/annotations/instances_test_verified.json
python scripts/calibration/make_folds.py --photos-coco <PackPhys instances_export.json> \
    --extra-coco beach=... --seed 2026            # main partition
python scripts/calibration/make_folds.py ... --seed 1 --out-data datasets/calibration/folds_rep1 \
    --out-cfg configs/folds_rep1 --data-root datasets/multimodal/v3.4_folds_rep1   # replicate
```

Both scripts are deterministic: the manifest, the yaml files and the generated
tuples reproduce bit for bit from the seed.

Licence: CC BY 4.0 (same as real495).
