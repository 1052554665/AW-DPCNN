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


### model training and evaluation
After preparing the fused dataset, the dataset was fed into the classification model for training. The training process involves the following steps:
1. Consider adding a **noise injection** or **cross-load** condition to make the CWRU benchmark more challenging and convincing
2. Load the fused dataset from `AW-DPCNN/dataset` and create data loaders for training, validation, and testing.
3. For the cross-severity experiment, train on ALL `cwru_raw_007` files and test on ALL `cwru_raw_014` files — this avoids leakage entirely since the recordings are physically different.
4. Some state-of-the-art classification models (e.g., ResNet, DenseNet, etc.) will be used as the backbone of the classification model. Those models can be referenced and revised in `src/models/`. The model will be trained using the training set, and the performance will be evaluated on the validation set to tune hyperparameters and prevent overfitting.
5. After training, the final model will be evaluated on the test set to assess its performance in terms of metrics such as accuracy, precision, recall, F1-score, F-measure, confusion matrix, ROC curve and AUC value.
6. t-SNE visualization will be performed to visualize the feature space and understand how well the model is separating different classes. The feature layers before the classifier head of the trained model will be used to extract features from the test set, and t-SNE will be applied to reduce the dimensionality for visualization.


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


