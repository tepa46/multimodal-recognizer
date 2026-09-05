# Data licences

The corpus is assembled from five sources plus our own, and they are not
uniformly licensed. This release reflects that: only material we may
redistribute ships as pixels; for the rest we release annotations together with
the scripts that reconstruct the benchmark from the upstream download.

| source | licence | shipped here |
|---|---|---|
| **real495** (ours) | CC BY 4.0 | images + annotations + measurements — in this repository |
| Drinking Waste | CC0 | annotations; images via `download_sources.py` |
| DWSD | CC BY 4.0 | annotations; images via `download_sources.py` |
| ZeroWaste-f | CC BY-NC 4.0 | annotations only |
| TACO | per-photo Flickr licences | annotations only |
| BeachLitter | research-only on some mirrors | our packaging labels only |

Notes:

- **ZeroWaste-f.** The MIT licence in the upstream repository covers the code,
  not the data; the dataset itself is CC BY-NC 4.0. Non-commercial use only.
- **TACO.** Every photograph carries its own Flickr licence. Images are obtained
  through the upstream downloader rather than redistributed here.
- **BeachLitter.** Held out of training entirely and used only as an
  out-of-distribution evaluation set. We release our own packaging labels
  (`results/ood/beachlitter_relabel/verified_labels.csv`) under CC BY 4.0; the
  underlying imagery remains under its original terms.
- **real495.** Photographed by the authors and published with their explicit
  consent. No people appear in any frame.

The proprietary 22-class production dataset referenced as previous work is not
part of this release and remains private. Every claim in the paper rests on the
public artefacts.

Reassembling the benchmark from upstream sources is the intended path:

```bash
python scripts/build_data/download_sources.py
python scripts/build_data/build_hybrid_coco.py --raw-dir datasets/raw/ \
    --output-dir datasets/processed_v3.3/ --exclude beach_litter \
    --zerowaste-cardboard-frac 0.33
```
