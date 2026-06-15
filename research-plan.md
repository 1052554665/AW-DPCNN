# Research plan
For the project `AW-DPCNN`, the research plan is divided into two main sections: supervised learning and unspervised learning. Each section will outline the steps and methodologies to be followed for dataset preparation, model training, and evaluation.

## for supervised learning

### CWRU Dataset Processing Strategy

Reviewing and revising the script `build_fused_dataset.py`, the script is responsible for building the fused dataset by combining the Mel spectrogram and GADF image using AW-DPCNN. The script likely performs the following steps:
1. Add `.mat` and `.wav` reading support directly to `build_fused_dataset.py`
2. Preprocess the data: Split at file level rather than segment level (e.g., 2 files train / 1 val / 1 test per class)
3. For each .mat file:
   a. Load `DE_time` signal by searching for keys containing `DE_time`
   b. Sliding-window segmentation, and segment each split independently (sampling rate: 12 kHz, window length: 2048 samples, hop length: 1024 samples)
   c. For each window:
      - Generate Mel spectrogram image, the parameters:
        - n_fft: 1024
        - n_mels: 128
        - fmax: 6000 Hz
        - image size: `224*224`
      - Generate GADF image, the parameters are as follows:
        - method: `difference`
        - image size: `224*224`
        - normalization range: `[-1, 1]`
        - colormap: `viridis`
      - Fuse via AW-DPCNN (γ=4, N=20)
      - Save as .png in split-appropriate class subfolder
4. Save `metadata.csv` (filename, class, source_file, window_idx, split)
5. Verify: per-class segment counts, JS divergence, feature-space MMD
6. The output is an ImageFolder-compatible directory tree (no further splitting needed)


The Normal (N) class has ~3.5× more windows than the fault classes because its recordings are longer. This is real-world class imbalance — your existing class-weighted loss in the training pipeline should handle it, but be aware of it when interpreting per-class metrics.


### Transformer Dataset Processing Strategy

Since the transformer data is not available, the next research will not consider real-world data of the transformer. There list the reason as follows:
1. The collection of transformer data should be succiffient, but the current data is not enough for training a deep learning model.
2. For supervised learning, the data should be labeled, but the current data is not all labeled.
3. For unsupervised learning, the data should be unlabeled. Consequently, in the later stages, the data can be divided into normal and abnormal data for unspervised learning.


### model training and evaluation
After preparing the fused dataset, the dataset was fed into the classification model for training. The training process involves the following steps:
1. Consider adding a **noise injection** or **cross-load** condition to make the CWRU benchmark more challenging and convincing
2. Load the fused dataset from `AW-DPCNN/dataset` and create data loaders for training, validation, and testing.
3. For the cross-severity experiment, train on ALL `cwru_raw_007` files and test on ALL `cwru_raw_014` files — this avoids leakage entirely since the recordings are physically different.
4. Some state-of-the-art classification models (e.g., ResNet, DenseNet, etc.) will be used as the backbone of the classification model. Those models can be referenced and revised in `src/models/`. The model will be trained using the training set, and the performance will be evaluated on the validation set to tune hyperparameters and prevent overfitting.
5. After training, the final model will be evaluated on the test set to assess its performance in terms of metrics such as accuracy, precision, recall, F1-score, F-measure, confusion matrix, ROC curve and AUC value.
6. t-SNE visualization will be performed to visualize the feature space and understand how well the model is separating different classes. The feature layers before the classifier head of the trained model will be used to extract features from the test set, and t-SNE will be applied to reduce the dimensionality for visualization.

About training strategy, there are some key design decisions to be made:
- **Epochs**: Start with 10 epochs for initial experiments, then increase to 30 epochs for more thorough training.
- **Early Stopping**: Implement early stopping based on validation loss to prevent overfitting and ensure that the model generalizes well to unseen data. Using a learning rate scheduler `ReduceLROnPlateau` with `patience=5` and early stopping with `patience=15` to adjust the learning rate dynamically based on the validation performance, which can help in achieving better convergence.
- **Learning Rate**: Start with a learning rate of 10⁻⁴, then decrease to 10⁻⁵ for fine-tuning after initial convergence.
- **Optimizer**: Use AdamW optimizer with a weight decay of 1e-3 to prevent overfitting and improve generalization. 
- **Batch Size**: Use a batch size of 32, which is a common choice for training deep learning models and should work well with the available computational resources.
- **Data Augmentation**: Apply data augmentation techniques (e.g., random cropping, horizontal flipping, color jittering) to increase the diversity of the training data and improve the model's robustness.
- **Evaluation Metrics**: In addition to accuracy, precision, recall, and F1-score, consider using additional metrics such as the confusion matrix, ROC curve, and AUC value to gain a more comprehensive understanding of the model's performance across different classes.
- **Cross-validation**: If computational resources allow, consider implementing k-fold cross-validation to further validate the model's performance and ensure that the results are robust across different subsets of the data.
- **Hyperparameter Tuning**: Use techniques such as grid search or random search to explore different combinations of hyperparameters (e.g., learning rate, batch size, weight decay) and identify the optimal settings for training the model.
- **Ensemble Methods**: Consider training multiple models with different architectures or hyperparameters and combining their predictions using ensemble methods (e.g., majority voting, weighted averaging) to potentially improve overall performance.
- **Model Interpretability**: Explore techniques for interpreting the model's predictions (e.g., Grad-CAM, SHAP values) to gain insights into which features are most important for classification and to ensure that the model is making decisions based on relevant information.
- **Reproducibility**: Ensure that the training process is reproducible by setting random seeds, documenting the training configuration, and saving the trained models and results for future reference and comparison.
- **Computational Resources**: Monitor the computational resources (e.g., GPU usage, memory) during training and adjust the batch size or model architecture if necessary to ensure efficient training without running into resource limitations.
- **Logging and Visualization**: Use tools such as TensorBoard or Weights & Biases to log training metrics, visualize the training process, and track the performance of different models and hyperparameter configurations over time.
- **Model Selection**: After training multiple models and configurations, select the best-performing model based on validation metrics and evaluate it on the test set to report the final results. Consider also reporting the performance of other models for comparison and to provide insights into the effectiveness of different architectures and training strategies.
- **Statistical Significance Testing**: If comparing multiple models or configurations, consider performing statistical significance testing (e.g., paired t-test) to determine whether observed differences in performance are statistically significant and not due to random chance.
- **Error Analysis**: After evaluating the model on the test set, perform an error analysis to identify common types of misclassifications and understand the limitations of the model. This can provide insights into potential areas for improvement and guide future research directions.


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