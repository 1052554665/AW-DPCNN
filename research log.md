# transformer-five dataset
>create a script same as `build_CWRU_dataset.py` to process the `raw-data/transformer-five` dataset, and make sure the data split is done at file level, not frame level, to avoid data leakage. The script should also include a verification step to check for any potential data leakage by comparing the distribution of classes in the training, validation, and test sets.

Here's the analysis of transformer-five:

| Class | Files | % |
|---|---|---|
| DCBias | 119 | 18.3% |
| Harmonic | 109 | 16.7% |
| Loosen | 119 | 18.3% |
| **Normal** | **185** | **28.4%** |
| PartialDischarge | 119 | 18.3% |
| **Total** | **651** | 100% |



## Script: build_transformer_five.py

### What it does
A single unified pipeline that reads **unsplit** `.wav` files from transformer-five, performs a **file-level** stratified split, and generates AW-DPCNN fused images — all in one step.

### Key design — File-level split (no leakage)

| vs | Old approach (`split_transformer_five.py` + build_transformer_dataset.py) | New approach (`build_transformer_five.py`) |
|---|---|---|
| Split granularity | Split .wav files first, then window → no leakage guarantee in build step | **File-level split inside the build pipeline** — all windows from a source file go to the same split |
| Verification | Manual | Automatic JS divergence check |

### Result

```
✅ 5,859 fused images generated
   train:  3,501 images  (DCBias:639  Harmonic:585  Loosen:639  Normal:999  PartialDischarge:639)
   val:    1,179 images  (DCBias:216  Harmonic:198  Loosen:216  Normal:333  PartialDischarge:216)
   test:   1,179 images  (DCBias:216  Harmonic:198  Loosen:216  Normal:333  PartialDischarge:216)

[OK] All 651 source files assigned to exactly one split — zero leakage
```

### Usage

```bash
# Default (recommended)
python scripts/build_transformer_five.py --workers 32

# Dry-run to preview the split
python scripts/build_transformer_five.py --dry-run

# Custom parameters
python scripts/build_transformer_five.py \
    --win-len 4096 --hop-len 1024 \
    --file-split 70,15,15 \
    --workers 16
```

# transformer-ten dataset
>Analyse the origin data `raw-data/Group2_4（original）` and create a script similar to `build_CWRU_dataset.py` to build the corresponding dataset. 
- The GX represents the group number, e.g., G2 represents group 2.
- Select the data from the original dataset, which includes 10 classes.
- The data spilting should obey the following principles:
  * The data should be split at file level, not frame level, to avoid data leakage.
  * Spilting the data with a proper ratio.
  * The distribution of classes in the training, validation, and test sets should be similar to ensure that the model is trained and evaluated on representative samples.
- The 10 classes in the original dataset are as follows, where Normal was represented by NoLoad.
  - Loosen
  - Overload
  - Normal
  - PartialDischarge
  - 30pThirdHarmonic
  - 30pFifthHarmonic
  - 30pSeventhHarmonic
  - pureFifthHarmonic
  - pureSeventhHarmonic
  - pureThirdHarmonic

The experiment results show that the validation accuracy is very low.

revise the script and the classes are as follows, where Normal was represented by NoLoad.
  - Overload
  - Normal
  - 10pThirdHarmonic
  - 10pFifthHarmonic
  - 10pSeventhHarmonic
  - 20FifthHarmonic
  - 20SeventhHarmonic
  - 20ThirdHarmonic

The validation accuracy is also low.



## Script: build_group2_4.py





### Key design

| Feature | Detail |
|---|---|
| **File-level split** | All windows from a source file go to exactly one split — zero leakage |
| **Duration-aware balancing** | Files sorted by window count, assigned via greedy balanced multi-way partition to minimize inter-split variance |
| **Verification** | JS divergence computed between all split pairs |
| **Group tracking** | `group` and `condition_raw` columns in metadata.csv |


### Usage

```bash
# Default
python scripts/build_group2_4.py --workers 32

# Dry-run
python scripts/build_group2_4.py --dry-run

# Custom split ratio
python scripts/build_group2_4.py --file-split 50,25,25 --workers 16
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


