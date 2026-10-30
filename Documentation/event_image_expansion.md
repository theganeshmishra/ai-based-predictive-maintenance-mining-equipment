# Expanded Pump and Haul Truck event images

The expanded experiment uses non-overlapping 125 ms polarity-count frames over up to 30 seconds of each available recording. It preserves the original recording groups. Additional frames are related observations, not additional machines or independent fault events. Both Pump archives passed publisher MD5 verification.

| Equipment mapping | Exported images | Held-out images | Held-out recordings | Image accuracy | Image macro F1 | Recording accuracy | Recording macro F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pump | 3,913 | 1,946 | 24 | 99.85% | 99.84% | 100% | 100% |
| Haul Truck | 2,691 | 1,392 | 12 | 56.32% | 52.93% | 58.33% | 56.55% |

Pump trains on 20 Hz recordings and tests on separate 30 Hz recordings. Haul Truck trains on 1000 rpm recordings and tests on separate 2000 rpm recordings. Epoch selection uses training recordings only. Test frames never enter training. Results describe separate laboratory component analogues, not deployment on mining fleets.

The expanded Haul Truck score is lower than the earlier 250 ms whole-recording result (75% recording accuracy). More frames did not improve generalization. Both experiments are retained under separate filenames; the expanded result must not overwrite or be presented as an improvement over the earlier result.

## Reproduce locally

With the source archives present and the project environment installed:

```sh
python Notebook/notebook_helpers.py pump export-images --dense-frames
python Notebook/notebook_helpers.py pump cnn-pretrained --dense-frames
python Notebook/notebook_helpers.py haul_truck export-images --dense-frames
python Notebook/notebook_helpers.py haul_truck cnn-pretrained --dense-frames
```

Exports are saved under `Images/<equipment>_denseframes/`. Results and figures use the `_denseframes` suffix. Raw archives, exported images, caches and model binaries remain ignored by Git.

The single notebook displays the number of held-out images in [the count chart](../Figures/test_image_counts.png). Confusion matrices and metrics are in the respective [Pump results](../Results/pump/pump_cnn_pretrained_4class_denseframes.json) and [Haul Truck results](../Results/haul_truck/haul_truck_cnn_pretrained_4class_denseframes.json). The original 24 Pump and 12 Haul Truck held-out recordings remain the independent evaluation units; image-level scores measure correlated frames.
