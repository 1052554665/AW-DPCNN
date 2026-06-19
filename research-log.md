>why there is noting change in the images before and after applying the parameter of `sequence_length`?

Good question. The issue is that `sequence_length` currently only controls how many raw samples go **into** the GAF transform, but `GramianAngularField` always reduces them to `img_size` (224) segments via PAA (Piecewise Aggregate Approximation) regardless. So whether you feed 8192 or 3000 samples, PAA averages them down to exactly 224 bins — the output is visually near-identical.

The fix is to make `sequence_length` control the **`image_size` parameter of `GramianAngularField`** itself (which sets the number of time segments in the GAF matrix), then resize back to `img_size`. This directly changes the GAF resolution.


| Function | `GramianAngularField(image_size=...)` | What `sequence_length` actually does |
|---|---|---|
| `generate_gadf_image` | Always `img_size` (224) | Pre-resamples signal → PAA still reduces to 224 bins |
| `_time_series_to_image` | Always `img_size` (224), baked into transformer | Same — pre-resamples, but PAA always → 224 bins |

**The fix**: Make `sequence_length` directly control `GramianAngularField`'s `image_size` (the number of time segments in the GAF matrix), then resize back to `img_size`. This changes the GAF resolution and will be visually apparent.

## Root cause & fix

**Before**: `sequence_length` only pre-resampled the raw signal before feeding it to `GramianAngularField(image_size=img_size)`. Since PAA inside `GramianAngularField` always reduced the signal to 224 bins regardless, the output was near-identical.

**After**: `sequence_length` now directly controls `GramianAngularField(image_size=sequence_length)` — i.e., the **number of time segments in the GAF matrix**. The resulting matrix is then resized to `img_size × img_size`. A lower value (e.g., 32) produces a coarse, blocky GAF; a higher value (e.g., 256) produces a fine-grained GAF.

```
sequence_length=32   →  32×32 GAF matrix  →  resize to 224×224  (coarse)
sequence_length=128  →  128×128 GAF matrix →  resize to 224×224  (medium)
sequence_length=None →  224×224 GAF matrix →  no resize          (default)
```

### Usage

```bash
# Coarse GAF (fast)
python scripts/build_group2_4_harmonic.py --sequence-length 32

# Fine-grained GAF (slower but more detail)
python scripts/build_group2_4_harmonic.py --sequence-length 256

# Default (same as --sequence-length 224)
python scripts/build_group2_4_harmonic.py
```