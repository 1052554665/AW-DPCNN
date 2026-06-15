>All comments from different journals for manuscript `AWDPCNN`.

# IEEE Sensor

## Reviewer: 1  

Recommendation: Reject (Paper is not acceptable for the Sensors Journal. Author should be encouraged to submit to another journal.)  

Comments:  
Upon reviewing this paper as a whole, it appears that most of the technical content involves incremental improvements to existing modules, without any particularly outstanding technical contributions. The following are suggestions for improvement:  
1. The current introduction of AW-DPCNN is primarily based on the premise that “PCNN possesses pulse-coupled characteristics, making it suitable for image fusion.” However, the paper does not clearly explain why PCNN is particularly well-suited for fusing these two heterogeneous acoustic representations—Mel and GADF. 
2. The current method comprises: Mel extraction, GADF encoding, AW-DPCNN fusion, and MSCA-VGG16 (including multi-scale convolutions, channel attention, and embedding heads). With too many components and insufficient decoupling of contributions across modules, readers may perceive this as a “package-style” innovation. 
3. The current baseline comparison includes basic CNNs, AlexNet, ResNet18, ConvNeXt-Tiny, and VGG16, but lacks models that have performed exceptionally well in the field of acoustic fault diagnosis in recent years (such as Transformer-based models, MobileNet, EfficientNet, or time-frequency fusion networks specifically designed for acoustic signals). 
4. Transformer-based models often struggle with acoustic signals in the field, which are frequently affected by environmental noise and interference sources. Although the paper uses field-collected data, it does not specifically evaluate noise robustness. 
5. GADF converts time series into images, which involves high computational complexity and is sensitive to signal length. The paper does not adequately explain why GADF was chosen over other time-domain encodings (such as Markov Transition Fields or recurrent graphs).

## Reviewer: 2  

Recommendation: Review Again After Resubmission (Paper is not acceptable in its current form, but has merit. A major rewrite is required. Author should be encouraged to resubmit a rewritten version after the changes suggested in the Comments section have been completed.)  
  
Comments:  
The manuscript addresses an important topic in acoustic-based power transformer fault diagnosis and proposes a multi-representation fusion framework combining Mel spectrograms, GADF images, AW-DPCNN fusion, and MSCA-VGG16 classification. The use of field transformer acoustic data is valuable. However, several issues should be addressed before the manuscript can be considered for publication.  

The manuscript combines several existing techniques, including PCNN/DPCNN, adaptive weighting, VGG16, multi-scale convolution, and channel attention. The authors should more clearly explain the specific methodological novelty of AW-DPCNN and MSCA-VGG16 compared with existing fusion and attention-based diagnosis methods. 

The manuscript uses Mel spectrograms and GADF images as two complementary acoustic representations. However, the reason for choosing these two image construction methods over other time-series/image representation methods is not sufficiently explained. The authors should provide a clearer physical or signal-processing motivation, explaining why Mel spectrograms and GADF are particularly suitable for transformer acoustic fault diagnosis. In addition, comparative experiments with other commonly used image construction methods, such as STFT spectrograms, CWT scalograms, GASF, recurrence plots, Markov transition fields, or MFCC-based representations, should be added under the same classifier and fusion settings to demonstrate the necessity and superiority of the selected representations. 

The adaptive weighting strategy based on local contrast appears heuristic. The authors should explain why this design is suitable for transformer acoustic signals and provide sensitivity analysis for key parameters such as the contrast amplification factor, local window size, and PCNN iteration number. 

There are inconsistencies between the text, equations, Fig. 4, and Table IV regarding the MSCA-VGG16 structure, especially the use of global average pooling, flattening, and fully connected layers. These descriptions should be carefully checked and unified. 

Please provide sufficient information about the data acquisition process, including microphone type, sampling settings, recording duration, number of sessions, environmental noise conditions, and label verification. Since overlapping segmentation is used, the authors should also clearly demonstrate that no data leakage occurs between training, validation, and test sets.  

The current results appear to be based on a fixed split. It is recommended to conduct repeated experiments with different random seeds or session-level splits and report mean ± standard deviation. More comparisons with representative acoustic fault diagnosis methods should also be included. 

The manuscript claims improved robustness under acoustic disturbances, but controlled noise experiments are limited. Additional tests under different SNR levels or environmental noise conditions would make the conclusions more convincing. Moreover, the CWRU dataset is a bearing vibration dataset rather than transformer acoustic data, so the generalization claim should be stated more cautiously. 

The additional validation on the CWRU dataset should be reconsidered or supplemented. Although CWRU is a widely used benchmark for bearing fault diagnosis, it is based on vibration signals rather than transformer acoustic signals, and therefore has limited relevance to the acoustic-based transformer diagnosis task studied in this manuscript. Since publicly available acoustic fault diagnosis datasets are now available, the authors are encouraged to conduct additional validation on acoustic datasets instead of relying mainly on the CWRU vibration dataset. This would provide more convincing evidence for the generalization capability of the proposed acoustic diagnosis framework. 

Since the proposed framework includes image transformation, AW-DPCNN fusion, and a VGG16-based network, the authors should report model parameters, FLOPs/MACs, inference time, and discuss whether the method is suitable for online transformer monitoring. 

Some figures are crowded or difficult to read, especially the graphical abstract, Fig. 1, Fig. 6, and the confusion matrices. Some table labels and figure captions contain typographical or consistency issues, such as “Traning,” “Loosenness,” and inconsistent class names. Several equations also need clearer formatting. 

The manuscript contains some awkward expressions, repetitive descriptions, and overstatements. Claims such as “superior robustness” and “generalization capability” should be supported by stronger evidence or written more conservatively. 

# Measurement
## Review 1
While it seems that the submission presents interesting results, the novelty of the proposed method is trivial. It only contains a combination of some new feature extraction and classfiers. The method indeed is still a traditional supervised learning-based fault diagnosis method. Major comments are as follows:

1. It can be seen in the literature that there are different types of such methods have been reported in this area. 
2. The experimental results are not sufficient. There is lack of ablation experiments， which are very important to validate the performance. 
3. The used dataset is not enough, the amount of the test data is too small to be valid. 
4. The involved parameters should be carefully optimized. And, the overfitting should be carefully solved. 
5. The proposed method should be compared with the popular "end-to-end" methods. 

## Review 2

Aiming at the problems of limited data and insufficient recognition accuracy in transformer fault diagnosis, this paper proposes a voiceprint diagnosis method based on Mel-GADF-PCNN and improved Physics-Informed Neural Networks (PINNs), which has certain theoretical value. However, there is still room for optimization in terms of innovation and technical details.

1. The abstract mentions using a Pulse-Coupled Neural Network (PCNN) to enhance discriminative features and suppress noise, while designing an improved AlexNet-SE architecture under the PINNs framework to achieve accurate fault diagnosis. Is the method of stacking multiple models reasonable? 

2. The proposed method only has certain innovations in data generation, but the improvements to the model are merely a permutation and combination of existing methods, lacking true innovation. 

3. The types of experimental data are too limited to prove the effectiveness of the method. 

4. The comparative methods are not representative; please supplement the latest models for comparison. 

5. Ablation experiments are lacking; please add them. 

6. When using deep learning models for intelligent recognition and classification, the input and output should be end-to-end. The method of generating new image data from raw signals via time-frequency diagrams and Gramian Angular Field encoding does not qualify as end-to-end and is therefore not recommended. 

7. The confusion matrix shown in Figure 12 is based on an insufficient amount of test data, and the classification performance appears to be unsatisfactory. 

## Review 3
The high experimental accuracy is overshadowed by potential data leakage risks and the use of outdated network architectures. Achieving 98.48% accuracy on a small dataset (520 samples) with a parameter-heavy model (AlexNet) strongly suggests overfitting or data leakage. The authors must clarify if the training and test sets were split by independent recording sessions or merely by slicing the same audio files.


## Review 4
3. More experimental data should be included to further substantiate the effectiveness of the proposed method. 

4. The baseline methods selected for comparison in this work are conventional; more recent state-of-the-art approaches should be incorporated to better demonstrate the comparative performance.  

5. The ablation studies conducted are insufficient, and the procedure for determining the hyperparameters has not been provided. 


## Review 5

The comments to the reviewers are given as below:

1. It requires extensive English revision. There are many grammatical mistakes and incorrect usage of vocabulary. For instance, "the Mel spectrogram transformers the original linear spectrogram into a logarithmic one." 

2. The overall Novelty of the paper is limited. All these techniques are existing techniques and are just applied. Can you elaborate what you have basically added to these structures? Similarly, the equations used are just general equations of these techniques and images.


5. "The resulting dataset was randomly split into training (80%), validation (10%), and test (10%) sets as shown in Table 3, to ensure a balanced and robust experimental setup." Generally, at least 20 to 25 % of the data is used for testing. Also "to ensure a balanced", however the number of samples in all the classes are different which results in an imbalanced dataset, how have you addressed this imbalance issue in your proposed method? “

6. You have not compared your method with any state-of-the-art method. You need to make at least 4 to 5 comparisons. 
# Other journals

## MSSP
We have put your manuscript on hold because you do not appear to have included references to the work that you are comparing your algorithm against.  
- Please can you ensure that your bibliography is updated to reflect accurate references for the following algorithm(s)/dataset(s): Baseline CNN, AlexNet, ResNet18, ConvNeXt-Tiny, VGG16, Case Western Reserve University Dataset

