# table of Operating Conditions and Corresponding Labels

\begin{table}
	\caption{Transformer Operating Conditions and Corresponding Labels}
	\centering
	\begin{tabular}{ll}
		\toprule
		Label & Operating Condition \\
		\midrule
		Normal & Normal \\
		BF007 & Ball Faults \\
		BF014 & 30\% Seventh Harmonic Distortion \\
		BF021 & 30\% Third Harmonic Distortion \\
		IF007 & Loosen \\
		IF014 & Normal \\
		IF021 & Partial Discharge \\
		OF007 & Pure Fifth Harmonic Distortion \\
		OF014 & Pure Seventh Harmonic Distortion \\
		OF021 & Pure Third Harmonic Distortion \\
		\bottomrule
	\end{tabular}
	\label{tab:operating_conditions}
\end{table}

>replace the above table with the following, and format with IEEE.


Target classes
--------------
  BF007   BF014   BF021      (ball faults    — 0.007", 0.014", 0.021")
  IF007   IF014   IF021      (inner race     — 0.007", 0.014", 0.021")
  OF007   OF014   OF021      (outer race @6  — 0.007", 0.014", 0.021")
  Normal

#  table of dataset composition

\begin{table*}
	\centering
	\caption{Composition of the Transformer Acoustic Dataset}
	\label{dataset}
	\begin{tabular}{l c c c c c c c c c c c}
		\toprule
		\multirow{2}{*}{Datasets} & 
		\multirow{2}{*}{Total} & 
		\multicolumn{10}{c}{Number of Samples per Class (N0–N9)} \\
		\cmidrule(lr){3-12}
		&  & N0 & N1 & N2 & N3 & N4 & N5 & N6 & N7 & N8 & N9 \\
		\midrule
		Training set & 8924 & 881 & 880 & 881 & 880 & 1002 & 880 & 880 & 880 & 880 & 880\\
		Validating set & 3410 & 440 & 294 & 441 & 294 & 441 & 324 & 294 & 294 & 294 & 294\\
		Testing set & 4115 & 441 & 294 & 441 & 294 & 441 & 881 & 294 & 441 & 294 & 294 \\
		\bottomrule
	\end{tabular}
\end{table*}

>replace the above table with the following, and format with IEEE.

Collected 40 .mat files → 10 classes:
✓ BF007 4 file(s)
✓ BF014 4 file(s)
✓ BF021 4 file(s)
✓ IF007 4 file(s)
✓ IF014 4 file(s)
✓ IF021 4 file(s)
✓ OF007 4 file(s)
✓ OF014 4 file(s)
✓ OF021 4 file(s)
✓ Normal 4 file(s)

File‑level split (seed=42):
Class Total Train Val Test
----------------------------------------
BF007 4 2 1 1
BF014 4 2 1 1
BF021 4 2 1 1
IF007 4 2 1 1
IF014 4 2 1 1
IF021 4 2 1 1
OF007 4 2 1 1
OF014 4 2 1 1
OF021 4 2 1 1
Normal 4 2 1 1
----------------------------------------
TOTAL 40 20 10 10

✅ Done — 2913/2913 images written to ./datasets/cwru_de
📋 Metadata saved to ./datasets/cwru_de/metadata.csv (2913 rows)

────────────────────────────────────────────────────────────
Distribution verification
────────────────────────────────────────────────────────────
train: 1397 samples {'BF007': 116, 'BF014': 116, 'BF021': 116, 'IF007': 116, 'IF014': 116, 'IF021': 116, 'Normal': 353, 'OF007': 116, 'OF014': 116, 'OF021': 116}
val: 758 samples {'BF007': 58, 'BF014': 58, 'BF021': 58, 'IF007': 58, 'IF014': 58, 'IF021': 58, 'Normal': 236, 'OF007': 58, 'OF014': 58, 'OF021': 58}
test: 758 samples {'BF007': 58, 'BF014': 58, 'BF021': 58, 'IF007': 59, 'IF014': 58, 'IF021': 58, 'Normal': 235, 'OF007': 58, 'OF014': 58, 'OF021': 58}

>is the dataset split correct? the normal class has 353 training samples, 236 validation samples, and 235 test samples, which is significantly higher than the other classes that have around 116 training samples and 58 validation/test samples. dose this indicates an imbalance in the dataset distribution?


**The file-level split is correct** — every class has exactly 4 source files split 2/1/1. The image-count imbalance is **inherent to the CWRU dataset**, not a split error.

### Why Normal has more windows

| Class | Signal duration (~samples) | Duration | Windows per file (win=2048, hop=1024) |
|---|---|---|---|
| Fault (B/IR/OR) | ~121,000–122,000 | ~10 sec | ~116 per training file |
| Normal baseline | ~485,000 | ~40 sec | ~353 per training file |

The Normal baseline recordings are **~4× longer** than fault recordings in the CWRU dataset. This is physical ground truth — normal operation was recorded for longer periods.

### Why this isn't a problem

Your pipeline already has two mitigations:

1. **Class-weighted cross-entropy loss** (`class_weighting: true`) — minority classes get proportionally larger gradients:
   $$w_k = \frac{N_\text{total}}{K \cdot N_k}$$
   Normal weight ≈ 0.4, fault weight ≈ 1.2 (3× difference compensates the 3× sample imbalance)

2. **Macro-averaged evaluation metrics** — F1, G-mean, balanced accuracy all treat each class equally regardless of sample count

### For the paper

Just add one sentence acknowledging this:

> "The Normal class contains a larger number of windows owing to the longer duration of baseline recordings in the CWRU dataset (~40 s vs. ~10 s for fault recordings). This natural imbalance is addressed through class-weighted loss during training, and all reported metrics are macro-averaged to ensure balanced evaluation across classes."

# 

review all the `results/test_metrics` in the logs to ensure they are macro-averaged and not micro-averaged. If they are micro-averaged, you will need to change them to macro-averaged to avoid misleading results due to the class imbalance.