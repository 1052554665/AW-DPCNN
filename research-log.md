In the confusion matrix, some of the values are 100%, review relative scripts, the following methods can be referred to fix the issue.


# 100% Values in Confusion Matrix: Red Flag Assessment

**Short answer**: It depends on *which* cells are 100%, but **a 100% value anywhere is a serious warning sign** that needs investigation before you trust your results.

## What 100% Means (By Position)

### **Diagonal (True Positives): 100% for a class**
Example: All Ball fault samples classified as Ball fault (0 misclassifications)

**Reasonableness**: 
- ❌ **Suspicious on test set** — extremely unlikely for real data unless:
  - Test set is tiny (≤5 samples/class) — variance, not signal
  - **Most likely**: Data leakage (train/test overlap, preprocessing fit on full data, signal windows split incorrectly)
  - Model overfit to test distribution
- ✓ **Acceptable on training set** — overfitting is expected if you train long enough
- ✓ **Possible on small hold-out sets** — but still investigate

**Diagnostic check**:
```python
# How many samples in each class on your test set?
print(test_labels.value_counts())
# If any class has <10 samples, 100% accuracy is just noise
```

### **Off-Diagonal (False Positives/Negatives): 100%**
Example: Every Inner Race sample misclassified as Ball fault (0 correct)

**Reasonableness**:
- ❌ **Almost never reasonable** — means the model *never* got that class right
- Suggests:
  - **Severe class confusion** (Inner Race features overlap with Ball fault more than any other class)
  - **Class imbalance handling failed** (minority class ignored by loss function)
  - **Data leakage created fake correlation** (e.g., Inner Race only appears in certain training conditions)

## How to Diagnose Leakage

Run a **label-shuffling sanity check** immediately:

```python
import numpy as np
from sklearn.metrics import confusion_matrix

# Save your current test predictions
y_pred_real = model(test_data)  # Your model's predictions

# Shuffle test labels randomly
y_test_shuffled = np.random.permutation(test_labels)

# Compute confusion matrix on shuffled labels
cm_shuffled = confusion_matrix(y_test_shuffled, y_pred_real, normalize='true')
print("Shuffled confusion matrix (should be ~25% diagonal for 4-class):")
print(cm_shuffled)

# If shuffled CM is also high on diagonal → LEAKAGE
# If shuffled CM drops to ~25% → no leakage, your model is just good
```

**Interpretation**:
- **Real model, no leakage**: Shuffled CM ~25% diagonal (random chance), real CM >80% diagonal
- **Leakage present**: Both real and shuffled CM show high diagonal (model exploiting something)


## Common Leakage Sources for Your CWRU Pipeline

Check these specifically:

### **1. Preprocessing fit on full dataset**
```python
# ❌ WRONG: Fits scaler on all data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_full)  # Full dataset!
X_train, X_test = train_test_split(X_scaled, ...)

# ✓ CORRECT: Fits scaler on train only
X_train, X_test = train_test_split(X_full, ...)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Only transform, don't fit
```

### **2. Mel spectrogram / GADF generation before splitting**
```python
# ❌ WRONG: Computes spectrograms on full dataset
mel_specs = [compute_mel_spec(signal) for signal in X_full]
X_train, X_test = train_test_split(mel_specs, ...)

# ✓ CORRECT: Split raw signals first, then compute features per set
X_train, X_test = train_test_split(X_signals, ...)
mel_train = [compute_mel_spec(s) for s in X_train]
mel_test = [compute_mel_spec(s) for s in X_test]
```

### **3. Signal windowing at wrong granularity**
If CWRU signals are ~1024 or 2048 samples and you use **sliding windows**, ensure:
```python
# ❌ WRONG: Overlapping windows can span train/test boundary
windows = [signal[i:i+256] for i in range(0, len(signal), 128)]  # 50% overlap
X_train, X_test = train_test_split(windows, ...)  # Windows from same signal split!

# ✓ CORRECT: Split at signal level, then window per set
train_signals, test_signals = train_test_split(signals, ...)
train_windows = [w for sig in train_signals for w in sliding_window(sig)]
test_windows = [w for sig in test_signals for w in sliding_window(sig)]
```

### **4. Data augmentation before split**
```python
# ❌ WRONG: Augments full dataset, then splits
X_aug = augment(X_full)  # Creates synthetic variations
X_train, X_test = train_test_split(X_aug, ...)

# ✓ CORRECT: Split first, augment train only
X_train, X_test = train_test_split(X_full, ...)
X_train_aug = augment(X_train)  # Synthetic data only in train
```

### **5. Hyperparameter tuning on test set**
```python
# ❌ WRONG: Using test set for model selection
for lr in [0.001, 0.01, 0.1]:
    model = train(X_train, y_train, lr)
    acc = evaluate(X_test, y_test)  # Tuning on test!
    if acc > best_acc:
        best_lr = lr

# ✓ CORRECT: Use validation set
X_train, X_val = train_test_split(X_train, test_size=0.15, ...)
for lr in [0.001, 0.01, 0.1]:
    model = train(X_train, y_train, lr)
    acc = evaluate(X_val, y_val)  # Tune on validation
    if acc > best_acc:
        best_lr = lr
# Final evaluation on held-out X_test
```

## What "Reasonable" 100% Looks Like

| Scenario | 100% Accuracy | Verdict |
|----------|---------------|---------|
| 1 sample/class in test set | 100% possible | **Noise**, ignore |
| 100 samples/class, 100% accuracy | Possible but rare | **Investigate leakage** |
| Training set, 100% accuracy | Expected | **Normal**, shows capacity |
| Validation set, 100% accuracy | Unlikely | **Likely leakage** |
| Test set, 100% accuracy | Very unlikely | **Almost certainly leakage** |

## What to Report

If you find 100% in your test confusion matrix:

1. **Run the label-shuffling check** above
2. **Audit your preprocessing pipeline** using the checklist above
3. **If no leakage found**: Report test set size per class and note that perfect accuracy may reflect small sample sizes rather than generalization
4. **If leakage found**: Fix it, retrain, and report corrected results



# Data Leakage Audit Report

### 🔴 CRITICAL Finding: Intra-Recording Chunk Leakage in build_group2_4_harmonic.py

**File**: build_group2_4_harmonic.py, function `_file_level_split_small()` (lines ~170-270)

**Root Cause**: 6 of 9 target classes have only **2 source `.wav` files** each (from groups G3, G4). When a class has `<3 files`, the code chunks the longest file into temporally-contiguous segments and distributes these chunks across **train/val/test splits**. Since chunks from the same recording share identical acoustic conditions (same microphone position, same environmental noise, same transformer state), the model learns to recognize recording-specific artifacts rather than fault signatures.

```python
# Lines 200-240 in build_group2_4_harmonic.py — THE LEAKAGE
if n < 3:
    # Chunks the longest file...
    chunks = []
    for i in range(n_chunks):
        start = i * chunk_len
        end = total_len if i == n_chunks - 1 else (i + 1) * chunk_len
        chunk_sig = sig[start:end].copy()
        chunks.append((chunk_path, sr, chunk_sig, group, condition))
    # ❌ Chunks go to different splits via greedy partition below
```

**Affected classes** (2 files each → chunking triggered):

| Class | Source Groups |
|---|---|
| 10pThirdHarmonic | G3, G4 |
| 10pFifthHarmonic | G3, G4 |
| 10pSeventhHarmonic | G3, G4 |
| 20pThirdHarmonic | G3, G4 |
| 20pFifthHarmonic | G3, G4 |
| 20pSeventhHarmonic | G3, G4 |

**Why this causes 100% accuracy**: The model memorizes recording-specific background noise patterns. A chunk in the test set from the same recording as a chunk in the training set will be trivially classified — the model has already "seen" almost identical acoustic conditions.

---

### 🟡 MINOR Finding: CWRU Fan-End Dataset Undersampling

**File**: build_cwru_fe.py

OF014 and OF021 have only **1 file** each in the fan-end dataset. This means `n_train=1, n_val=0, n_test=0` — all samples go to training, making validation/test evaluation impossible for these classes.

---

### ✅ No Issues Found

| Check | Status |
|---|---|
| Normalization fit on full dataset | ✅ Uses fixed ImageNet stats (mean/std), no data-dependent leakage |
| Mel/GADF before split | ✅ Split at file level first, then windows generated per split |
| Augmentation before split | ✅ `train_tf` (with augmentation) only applied to training set; `eval_tf` for val/test |
| Hyperparameter tuning on test | ✅ Early stopping monitors `val_f1`; test set only evaluated at end |
| Window overlap across splits | ✅ Windows from a file all go to exactly one split (file-level integrity) |

---

### Recommended Fixes

**Fix 1 (Primary)**: Replace chunking with session-level split in build_group2_4_harmonic.py. Instead of chunking files to populate all splits, assign entire recordings (or entire groups) to splits:

```python
# Group-level split: G2→train, G3→val, G4→test
GROUP_TO_SPLIT = {"2": "train", "3": "val", "4": "test"}
```

This guarantees zero acoustic overlap between splits. For CWRU, the 4 load-condition files are already independent recordings, so the existing `_file_level_split` is sufficient.

**Fix 2**: For the chunking fallback (when it's truly unavoidable), group all chunks from the same source file into the **same split** rather than distributing them:

```python
# Instead of distributing chunks across splits, assign 
# all chunks from file X to a single randomly-chosen split
```

**Fix 3**: Add a label-shuffling sanity check to the evaluation pipeline (evaluate.py or workflow.py) to automatically detect leakage:

```python
# If shuffled accuracy >> 1/K, leakage is present
shuffled_acc = accuracy(np.random.permutation(y_true), y_pred)
if shuffled_acc > 1/num_classes + 0.05:
    print(f"[WARN] Possible data leakage: shuffled acc = {shuffled_acc:.4f}")
```


# Add a label-shuffling sanity check

### Changes Made

**1. train_eval.py** — New function `check_label_leakage()`

| Parameter | Default | Purpose |
|---|---|---|
| `y_true`, `y_pred` | — | Ground-truth and predicted labels |
| `num_classes` | — | Number of classes |
| `num_shuffles` | 5 | Shuffle trials for stable estimate |
| `threshold` | 0.10 | Accuracy margin above chance to trigger warning |

Returns a dict with:
- `shuffled_acc` — mean accuracy over shuffled trials
- `chance_level` — expected random accuracy (1/K)
- `is_suspicious` — `True` if margin > threshold
- `warning` — human-readable message

**2. workflow.py** — Runs check after test evaluation
- Prints `[OK]` with stats when clean
- Prints `!!!! DATA LEAKAGE WARNING !!!!` when suspicious
- Records `shuffled_acc`, `chance_level`, `leakage_suspicious` in results JSON

**3. evaluate.py** — Same check for standalone evaluation

### How It Works

```
After model evaluation:
  1. Shuffle y_true labels randomly (5 trials)
  2. Compute accuracy with shuffled labels
  3. Compare against expected chance level (1/K)
  
  shuffled_acc ≈ 1/K   →  ✓ No leakage
  shuffled_acc >> 1/K   →  ⚠ LEAKAGE DETECTED
```

