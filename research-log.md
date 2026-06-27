>OF014 and OF021 show 0 val/test files in the split output, even though the raw data exists for those classes. This is because OF014 and OF021 only have 1 .mat file each in the FE dataset.

This isn't a bug — it's a fundamental data limitation. Let me explain and provide the best fix:

**Root cause**: OF014 and OF021 each have only **1 `.mat` file** in the FE dataset. With file-level 60/20/20 split, a single file goes entirely to one split (train). No files remain for val/test.

**The fix**: For single-file classes, split the segmented **windows** across splits instead of the file.


✅ Fixed. OF014 and OF021 now each show **1/1/1** across train/val/test. 

**How it works**: For single-file classes, the same `.mat` recording is assigned to all three splits, but windows are taken from **non-overlapping time regions**:
- Train: 0–60% of the signal
- Val: 60–80%  
- Test: 80–100%

This avoids data leakage while ensuring all 10 classes appear in every split.

Made changes.



>whether the time-region splitting approach for single-file classes is methodologically correct. 

It's a pragmatic compromise — **acceptable but with caveats**. Here's the honest assessment:

---

### What the method does

```
Single .mat file for OF014 (e.g., 10 sec @ 12 kHz ≈ 120,000 samples)

├── Train:  samples [0 ...... 72,000)     → ~35 windows (60%)
├── Val:    samples [72,000 .. 96,000)    → ~11 windows (20%)
└── Test:   samples [96,000 .. 120,000]   → ~11 windows (20%)
```

---

### ✅ What's right

| Aspect | Verdict |
|---|---|
| **No duplicate windows** across splits | ✅ Different time regions, no overlap |
| **All 10 classes represented** in val/test | ✅ Enables full evaluation |
| **Windows are from the same operating condition** | ✅ Consistent with train |

### ⚠️ What's imperfect

| Concern | Severity |
|---|---|
| **Reduced independence** — same recording, same sensor, same load | ⚠️ Val/test accuracy on OF014/OF021 may be **optimistic** (easier than a truly independent recording) |
| **Low sample count** — ~11 test windows per class | ⚠️ Barely enough for stable metrics |
| **Non-stationarity risk** — if fault signal changes over the 10s recording, the 60/20/20 time split is arbitrary | ⚠️ Minor for constant-speed bearings |

---

### Comparison with standard practice

| Method | Independence | All classes? | Publication-safe? |
|---|---|---|---|
| **File-level split** (what DE datasets use) | ✅ Different recordings | ✅ (4 files/class) | ✅ Gold standard |
| **Time-region split** (our fix for 1-file classes) | ⚠️ Same recording | ✅ | ⚠️ Acceptable with note |
| **Omit from val/test** | N/A | ❌ Missing 2 classes | ⚠️ Incomplete evaluation |
| **Duplicate windows** (same windows in train+test) | ❌ Pure leakage | ✅ | ❌ Rejected |

---

### Recommendation

**Keep the current approach**, but add one sentence to the paper:

> *"For the 12k FE dataset, fault classes OF014 and OF021 each contain a single recording. For these two classes, validation and test windows are drawn from non-overlapping temporal segments of the same recording (60%/20%/20% time-region split), while all other classes use independent file-level splits."*

This is honest, transparent, and reviewers will accept it as a reasonable handling of a dataset limitation. The DE datasets (12k/48k) have 4 files per class → no caveats needed there.


>whether it's correct that the 48k DE dataset uses different window/hop parameters (8192/4096) compared to 12k DE and 12k FE (2048/1024)

**Yes, it's correct — and by design.** All three datasets maintain the same **temporal** characteristics; only the sample counts differ due to the 4× higher sampling rate:

| Parameter | 12k DE / 12k FE | 48k DE | Ratio |
|---|---|---|---|
| **Sample rate** | 12,000 Hz | 48,000 Hz | 4× |
| **Window (samples)** | 2,048 | 8,192 | 4× |
| **Window (ms)** | **170.7 ms** | **170.7 ms** | ≡ |
| **Hop (samples)** | 1,024 | 4,096 | 4× |
| **Hop (ms)** | **85.3 ms** | **85.3 ms** | ≡ |
| **n_fft** | 1,024 | 4,096 | 4× |
| **Freq resolution** | **11.7 Hz/bin** | **11.7 Hz/bin** | ≡ |
| **fmax** | 6,000 Hz | 20,000 Hz | — |

The key design principle: **each fused image sees the same ~171 ms of bearing vibration**, regardless of sampling rate. This ensures the model learns fault signatures from equivalent physical time spans. The only difference is the 48k images capture richer high-frequency content (up to 20 kHz vs 6 kHz), which is the whole point of the cross-sampling-rate generalization experiment.
