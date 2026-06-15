# Research plan
For the project `AW-DPCNN`, the research plan is divided into two main sections: supervised learning and unspervised learning. Each section will outline the steps and methodologies to be followed for dataset preparation, model training, and evaluation.

## for supervised learning

### dataset preparation

Reviewing and revising the script `build_fused_dataset.py`, the script is responsible for building the fused dataset by combining the Mel spectrogram and GADF image using AW-DPCNN. The script likely performs the following steps:
1. Preprocess the data
   - **Split at file level rather than segment level** (e.g., 2 files train / 1 val / 1 test per class); 
   - **Segment each split independently** with your sliding window parameters
   - Extend the script to support `.mat` input directly.
2. Transform each waveform into the Mel spectrogram and GADF image using appropriate libraries and techniques (e.g., librosa for Mel spectrogram, and a suitable method for GADF image generation).
3. Combine the Mel spectrogram and GADF image into a single fused representation using the AW-DPCNN architecture. This may involve feeding both inputs into the model and extracting features to create a unified representation.
4. After generating the fused images independently for each split, verify that no file-level leakage occurred.
5. Save the fused dataset to `AW-DPCNN/dataset` for further use.
6. Compute class distribution statistics (imbalance ratio, entropy) across train/validation/test splits to confirm stratifications; verify no significant distribution shift between splits using JS divergence on label proportions.

The structure of the fused dataset is as follows:
- `AW-DPCNN/dataset/`
  - `train/`
    - `B`
      - `001.png`
      - `002.png`
      - ... 
    - `IR`
      - `001.png`
      - `002.png`
      - ...
    - `OR` 
      - `001.png`
      - `002.png`
      - ...
    - `N`
      - `001.png`
      - `002.png`
      - ...
  - `validation/`
    - `B`
      - `001.png`
      - `002.png`
      - ... 
    - `IR`
      - `001.png`
      - `002.png`
      - ...
    - `OR` 
      - `001.png`
      - `002.png`
      - ...
    - `N`
      - `001.png`
      - `002.png`
      - ...
  - `test/`
    - `B`
      - `001.png`
      - `002.png`
      - ... 
    - `IR`
      - `001.png`
      - `002.png`
      - ...
    - `OR` 
      - `001.png`
      - `002.png`
      - ...
    - `N`
      - `001.png`
      - `002.png`
      - ...
  - `metadata.csv` (optional, containing labels and other relevant information)


For the `transformer` dataset, the following steps will be taken:
- check the directoy tree of the folder `AW-DPCNN/raw-data/transformer`, refer to the script `build_fused_dataset.py` to create a script for `transformer` dataset.
- the `transformer` dataset had been spilted, consequently, do not need to split the dataset again, just transform the waveform into the Mel spectrogram and GADF image, and fused them via AW-DPCNN, and save the fused image into `AW-DPCNN/dataset/transformer/` with the same directory structure as above.


---

### Script: build_transformer_dataset.py

| Property | Value |
|---|---|
| Sample rate | 44,100 Hz |
| Duration per file | ~1 s (44,100 samples) |
| Classes | 10 (Loosen, Normal, PartialDischarge, 10kvOverload, 30pThirdHarmonic, 30pFifthHarmonic, 30pSeventhHarmonic, pureThirdHarmonic, pureFifthHarmonic, pureSeventhHarmonic) |
| Split | Pre-split: train (~1023 files) / val (~363 files) / test (~363 files) |
| Structure | `raw-data/transformer/{train,val,test}/{class_name}/*.wav` |

#### Key design decisions

| Aspect | Detail |
|---|---|
| **Input** | `raw-data/transformer/{train,val,test}/{ClassName}/*.wav` (pre-split, honoured exactly) |
| **Output** | `datasets/transformer/{train,val,test}/{ClassName}/*.png` (ImageFolder-compatible) |
| **No re-splitting** | The existing train/val/test partition is preserved — no file-level split |
| **Fusion** | Same AW-DPCNN algorithm (`γ=4`, `N=20`) as `build_fused_dataset.py` |

#### Dry-run results

| Split | Windows (images) |
|---|---|
| train | 40,920 |
| val | 14,520 |
| test | 14,520 |
| **Total** | **69,960** |

All 10 classes present in each split. Output paths follow the pattern:  
`datasets/transformer/train/10kvOverload/G4_10kvOverload_seg0_00000.png`

#### To run

```bash
# Enter the project root directory
cd AW-DPCNN
# Default parameters (recommended)
python scripts/build_transformer_dataset.py --workers 16 --metadata --verify

# Custom window / Mel parameters
python scripts/build_transformer_dataset.py \
    --win-len 4096 --hop-len 1024 \
    --n-fft 2048 --n-mels 128 --fmax 8000 \
    --workers 16 --metadata --verify
```

>according to the part of  model training and evaluation,  check and revise all the configuration files, models.

### model training and evaluation

- **Evaluation Metrics**: Adding additional metrics such as the ROC curve, and AUC value to gain a more comprehensive understanding of the model's performance across different classes.
- **Epochs**: Start with 10 epochs for initial experiments, then increase to 30 epochs for more thorough training.
- **Early Stopping**: Implement early stopping based on validation loss to prevent overfitting and ensure that the model generalizes well to unseen data. Using a learning rate scheduler `ReduceLROnPlateau` with `patience=5` and early stopping with `patience=15` to adjust the learning rate dynamically based on the validation performance, which can help in achieving better convergence.
- **Learning Rate**: Start with a learning rate of 10⁻⁴, then decrease to 10⁻⁵ for fine-tuning after initial convergence.
- **Optimizer**: Use AdamW optimizer with a weight decay of 1e-3 to prevent overfitting and improve generalization. 
- **Batch Size**: Use a batch size of 32, which is a common choice for training deep learning models and should work well with the available computational resources.
- **Data Augmentation**: Apply data augmentation techniques (e.g., random cropping, horizontal flipping, color jittering) to increase the diversity of the training data and improve the model's robustness.

---

Here's the analysis of what needs to change:

| Area | Current | Required |
|---|---|---|
| **Scheduler** | `StepLR` / `CosineAnnealingLR` | `ReduceLROnPlateau` (patience=5) |
| **Early Stopping** | ❌ Not implemented | patience=15 on val_f1 |
| **ROC/AUC** | ❌ Not implemented | Add to metrics & evaluation |
| **Augmentation** | RandomFlip, RandomRotation | Add RandomCrop, ColorJitter |
| **MA_CBAM_Resnet50.yaml** | Empty | Fill with proper config |
| **Several exp configs** | `cosine` scheduler | → `plateau` |



#### Summary of Changes

##### 1. metrics.py — Added ROC-AUC
- Added `compute_roc_auc(y_true, y_score, num_classes)` using `sklearn.metrics.roc_auc_score` with one-vs-rest multi-class support

##### 2. train_eval.py — Probabilities + AUC helper
- `evaluate()` now returns **5 values**: `(loss, metrics, y_true, y_pred, y_score)` — the 5th is softmax probabilities for AUC
- Added `calculate_roc_auc()` helper function

##### 3. workflow.py — Scheduler + Early Stopping + AUC
| Feature | Before | After |
|---|---|---|
| **Scheduler** | `StepLR` only | `ReduceLROnPlateau` (mode=max, patience=5, factor=0.5) + legacy `StepLR`/`Cosine` still supported |
| **Early Stopping** | ❌ | ✅ patience=15 on `val_f1` (configurable) |
| **Best model** | Saved by F1 | Saved by F1, tracked with epoch number |
| **val_auc** | ❌ | ✅ Logged every epoch |
| **test_auc** | ❌ | ✅ In final results JSON |
| **Checkpoint restore** | Always loads best.pt | Guarded with existence check |

##### 4. image_classification.py — Augmentation
- Added `RandomResizedCrop(scale=(0.8, 1.0))` 
- Added `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)`

##### 5. evaluate.py — Updated for new API
- Handles new 5-tuple return from `evaluate()`
- Prints `test_auc` alongside other metrics

##### 6. default.yaml — Updated defaults
```yaml
train:
  epochs: 30          # was 10
  early_stopping:
    enabled: true
    patience: 15
    metric: val_f1

scheduler:
  type: plateau       # was step
  mode: max
  factor: 0.5
  patience: 5
  min_lr: 0.000001

visualization:
  roc_curve: true     # new
```

##### 7. All 17 experiment configs — Standardized
| Change | Detail |
|---|---|
| Scheduler | `cosine` → `plateau` with `mode:max, factor:0.5, patience:5, min_lr:1e-6` |
| Epochs | `10` → `30` (baseline, alexnet_se, MSCA_VGG16) |
| `num_classes` | Fixed from `10` → `4` (CWRU dataset has 4 classes) |
| `lr` | Fixed `1e-5` → `1e-4` (baseline, MSCA_VGG16 which were fine-tuning too early) |
| MA_CBAM_Resnet50 | Was empty → filled with proper config |


### add extra metrics
>also add extra metrics: parameters and FLOPs, and save the ROC curve plot for each model for visualization and comparison.

#### Changes Summary

##### 1. metrics.py — Model complexity
- **`count_parameters(model)`** — counts trainable parameters
- **`compute_flops(model, input_shape, device)`** — estimates MACs via forward hooks on Conv2d, Linear, BatchNorm2d, ReLU, pooling layers. No external dependencies needed.

##### 2. plot_roc.py — New file
- **`plot_roc_curves(y_true, y_score, class_names, save_path)`** — plots per-class ROC curves (one-vs-rest) plus micro/macro average, with per-class AUC in the legend. IEEE-style formatting.

##### 3. workflow.py — Integration
- Computes **params & FLOPs** at training start, printed to console
- Saves **ROC curve** to `figures/roc_curve.png` after testing
- Includes `params` and `flops` in the results JSON

##### 4. evaluate.py — Integration
- Same params/FLOPs/ROC additions for standalone evaluation

##### Output per run
| File | Content |
|---|---|
| `figures/roc_curve.png` | Per-class + micro/macro ROC curves |
| `results/test_metrics.json` | Includes `params`, `flops` fields |
| Console | `Params: X.XXM \| FLOPs: X.XXM` |

>always print the tool name alongside the numbers for clarity, e.g.:
```
EfficientNet-B0 | Params: 5.3M | FLOPs: 780M
```


### Comparison experiments
To evaluate the effectiveness of the AW-DPCNN architecture, comparison experiments will be conducted using different fusion methods and model architectures. This will involve:
- Implementing alternative fusion methods (e.g., early fusion, late fusion, and other feature-level fusion techniques) and comparing their performance against the AW-DPCNN approach.

### Ablation studies
Ablation studies will be conducted to understand the contribution of different components of the AW-DPCNN architecture. For each ablation, create a configuration file that specifies which components are removed or modified. This will involve:
- Removing multi-scale, channel attention, and embedding head, separately and evaluating the impact on performance to identify which components are most critical for the model's success.
- Analyzing the results of the ablation studies to gain insights into the model's behavior and identify potential areas for improvement.






## for unsupervised learning (disconnected and not included in the current version of the paper)

### dataset preparation
To implement an unsupervised system using this dataset, the following steps will be taken:
### 1. Data Preparation
For Toy conveyor datasets, it will be processed as follows, you can refer those scripts in `E01_simple_AE_test/torch_version/` for more details.
- Training Set: Use only normal operating sounds. 1,000 randomly selected samples of individual (IND) normal sounds was used for training in `exp1_dataset_ToyConveyor/train_normal/`.
- Evaluation Set: Use the remaining IND normal samples and IND anomalous samples for testing in `exp1_dataset_ToyConveyor/test_normal/` and `exp1_dataset_ToyConveyor/test_anomalous/`.
- Preprocessing:
    * Downsampling: Downsample all audio to 16 kHz.
    * Noise Mixing: Mix the target sounds with environmental noise samples included in the dataset to simulate a real-world factory environment.
    * Feature Extraction: Same as the supervised learning, extract Mel spectrograms and GADF images from the audio samples and then fuse them with AW-DPCNN to create a fused representation.
    * Frame Concatenation: Concatenate the 10 frames before and after each frame (total 21 frames) to provide temporal context for the model.

### 2. Model Architecture (Autoencoder)
Refer to the scripts in `E01_simple_AE_test/torch_version/models.py`, some state-of-the-art autoencoder architectures (e.g., convolutional autoencoders, variational autoencoders, etc.) will be implemented and evaluated. The specific architecture will be chosen based on its suitability for modeling the normal sound patterns in the dataset.

### 3. Training and Inference
- training Objective: The autoencoder will be trained to minimize the reconstruction error of the normal training samples.
- Optimization: 
    * Start with a learning rate of 10⁻⁴.
    * Fix the rate for the first 25 epochs, then decrease it linearly to 10⁻⁶ over the next 25 epochs.
    * Conclude training after 50 epochs.
  * Anomaly Detection: 
    * Calculate the reconstruction error for each time frame of the test files.
    * A file is determined to be anomalous if the anomaly score exceeds a defined threshold for even one frame. The construction error of the AE was used as the anomaly score, and the parameters of the AE were trained to minimize the anomaly score of normal training samples.
* Evaluation metrics for the AE: 
    - Area Under the Receiver Operating Characteristic Curve (AUC-ROC): This metric evaluates the model's ability to distinguish between normal and anomalous samples across different threshold settings.
    - Precision, Recall, and F1-score: These metrics will be calculated at a specific threshold to evaluate the model's performance in terms of correctly identifying anomalies (precision) and capturing all anomalies (recall).


